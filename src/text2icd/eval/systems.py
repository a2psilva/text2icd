"""Systems the evaluation runner can score: the baselines, and the text2icd resolver as it is
built (`index`: the Alphabetic Index walk alone; `coder`: the walk and Tabular verification).

A system turns one phrase into a `Prediction` (the code set, ranked alternates, a confidence and
what it cost). It never sees the gold codes. Systems are built by name from `SYSTEMS`, with
string options from the command line (`-O k=20`).
"""

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import ClassVar

from pydantic import JsonValue

from text2icd.config import Settings
from text2icd.embeddings import Embedder, StaticEmbedder, VectorIndex
from text2icd.eval.predictions import Prediction
from text2icd.graph.model import Graph
from text2icd.llm.batch import BatchError, BatchModel, Collector, Replay, answer_in_rounds
from text2icd.llm.cost import CostMeter, PriceError
from text2icd.llm.navigator import LLMNavigator, Navigator, build_navigator
from text2icd.llm.prompts import PROMPT_VERSION
from text2icd.llm.providers import LLMUnavailableError, MessagesModel
from text2icd.resolve.coder import CodeSet
from text2icd.resolve.confidence import CalibrationError, read_calibration
from text2icd.resolve.resolver import Resolver, ResolverError
from text2icd.resolve.tabular import EPISODES, Episode
from text2icd.resolve.walk import WIDTH, IndexWalker
from text2icd.text.model import Setting

Options = Mapping[str, str]


class SystemSetupError(ValueError):
    """A system that cannot be built with the given options."""


class System(ABC):
    name: ClassVar[str]
    uses_llm: bool = False  # sends phrases to an LLM (a class may set it per instance)

    @abstractmethod
    def predict(self, phrase: str) -> Prediction:
        """The prediction for one phrase; the runner fills in the id and, if left at 0, the
        latency."""

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        """Predictions in order. Systems that work in batches override this."""
        for phrase in phrases:
            yield self.predict(phrase)

    def settings(self) -> dict[str, JsonValue]:
        """What the result depends on besides the code (model ids, k), for provenance."""
        return {}

    @property
    def endpoint(self) -> str | None:
        """The hosted service the system sends phrases to, or None if it runs locally."""
        return None

    def warnings(self) -> list[str]:
        """What the run should report besides its metrics (an LLM prompt cache never read)."""
        return []

    def close(self) -> None:  # noqa: B027  (optional hook)
        """Release what the system holds (clients, models)."""


def option(options: Options, key: str, default: int) -> int:
    try:
        return int(options.get(key, default))
    except ValueError:
        raise SystemSetupError(f"option {key} must be an integer, got {options[key]!r}") from None


class BM25System(System):
    """B1, the lexical floor: BM25 over Tabular descriptions and inclusion terms (the graph's
    full-text index), keeping billable codes in rank order. The confidence grows with the score
    gap between the first two codes."""

    name = "bm25"

    def __init__(self, graph: Graph, options: Options) -> None:
        self.graph = graph
        self.k = option(options, "k", 50)
        self.top = option(options, "top", 5)
        unknown = set(options) - {"k", "top"}
        if unknown:
            raise SystemSetupError(f"unknown options for bm25: {', '.join(sorted(unknown))}")

    def predict(self, phrase: str) -> Prediction:
        ranked: dict[str, float] = {}
        for hit in self.graph.search_tabular(phrase, self.k):
            if hit.node not in ranked and self.graph.code(hit.node).billable:
                ranked[hit.node] = hit.score
            if len(ranked) == self.top:
                break
        codes = list(ranked)
        if not codes:
            return Prediction(id="", confidence=0.0)
        scores = list(ranked.values())
        gap = scores[1] - scores[0] if len(scores) > 1 else 10.0
        return Prediction(
            id="",
            codes=(codes[0],),
            alternates=tuple(codes[1:]),
            confidence=1 - math.exp(-max(gap, 0.0)),
        )

    def settings(self) -> dict[str, JsonValue]:
        return {"k": self.k, "top": self.top, "tokenizer": self.graph.meta.get("fts_tokenizer")}


DEFAULT_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings-8M"
_QUERY_BATCH = 1024


class EmbeddingSystem(System):
    """B2, the semantic floor at zero API cost: the nearest Tabular descriptions and inclusion
    terms (the same texts as B1) by cosine similarity of local embeddings, keeping billable
    codes in rank order. The confidence is the similarity of the first code's text."""

    name = "embedding"

    def __init__(self, graph: Graph, options: Options, embedder: Embedder | None = None) -> None:
        self.graph = graph
        self.k = option(options, "k", 50)
        self.top = option(options, "top", 5)
        unknown = set(options) - {"k", "top", "model"}
        if unknown:
            raise SystemSetupError(f"unknown options for embedding: {', '.join(sorted(unknown))}")
        self.embedder = embedder or StaticEmbedder(options.get("model", DEFAULT_EMBEDDING_MODEL))
        texts = list(graph.tabular_texts())
        self.nodes = [text.node for text in texts]
        self.index = VectorIndex(self.embedder.encode([text.text for text in texts]))

    def predict(self, phrase: str) -> Prediction:
        return next(self.predict_many([phrase]))

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        for start in range(0, len(phrases), _QUERY_BATCH):
            batch = phrases[start : start + _QUERY_BATCH]
            began = time.perf_counter()
            hits = self.index.search(self.embedder.encode(batch), self.k)
            each = (time.perf_counter() - began) * 1000 / max(len(batch), 1)
            for found in hits:
                yield self._prediction(found, each)

    def _prediction(self, hits: list[tuple[int, float]], latency_ms: float) -> Prediction:
        ranked: dict[str, float] = {}
        for row, similarity in hits:
            node = self.nodes[row]
            if similarity <= 0:
                break
            if node not in ranked and self.graph.code(node).billable:
                ranked[node] = similarity
            if len(ranked) == self.top:
                break
        if not ranked:
            return Prediction(id="", confidence=0.0, latency_ms=latency_ms)
        codes = list(ranked)
        return Prediction(
            id="",
            codes=(codes[0],),
            alternates=tuple(codes[1:]),
            confidence=min(max(ranked[codes[0]], 0.0), 1.0),
            latency_ms=latency_ms,
        )

    def settings(self) -> dict[str, JsonValue]:
        return {"model": self.embedder.model, "k": self.k, "top": self.top}


class IndexSystem(System):
    """The Index walk alone (`resolve.walk`), before Tabular verification: for each diagnosis
    of the phrase its first candidate (with the manifestation code, if any), then any external
    cause code the phrase calls for, and the other candidates of the first diagnosis as
    alternates. Codes may be partial or not billable until the Tabular List completes them
    (Phase 6). No confidence: the coder's needs the completed code set."""

    name = "index"

    def __init__(self, graph: Graph, options: Options) -> None:
        self.width = option(options, "width", WIDTH)
        setting = options.get("setting", "inpatient")
        unknown = set(options) - {"width", "setting"}
        if unknown:
            raise SystemSetupError(f"unknown options for index: {', '.join(sorted(unknown))}")
        if setting not in ("inpatient", "outpatient"):
            raise SystemSetupError(
                f"option setting must be inpatient or outpatient, got {setting!r}"
            )
        self.setting = setting
        self.walker = IndexWalker.from_graph(
            graph, "outpatient" if setting == "outpatient" else "inpatient", self.width
        )

    def predict(self, phrase: str) -> Prediction:
        result = self.walker.walk(phrase)
        codes: list[str] = []
        for walk in result.diagnoses:
            if walk.best is not None:
                codes.append(walk.best.code)
                if walk.best.manifestation:
                    codes.append(walk.best.manifestation)
        for walk in result.diagnoses:  # external causes after the diagnoses (I.C.20)
            if walk.external_causes:
                codes.append(walk.external_causes[0].code)
        first = result.diagnoses[0].candidates[1:] if result.diagnoses else ()
        return Prediction(
            id="",
            codes=tuple(dict.fromkeys(codes)),
            alternates=tuple(dict.fromkeys(candidate.code for candidate in first)),
        )

    def settings(self) -> dict[str, JsonValue]:
        return {"width": self.width, "setting": self.setting}


class CoderSystem(System):
    """The deterministic coder (L1, `resolve.coder`): the Index walk, then Tabular verification,
    so every code is billable. The code set is the coder's sequence: the diagnoses with their
    manifestations and companion codes, in the order the notes give, then the external causes;
    the alternates, the other completed candidates of the first diagnosis; the confidence, that
    of the calibration table (`resolve.confidence`), if one is shipped or configured.

    `-O retrieval=on` adds L2 (task 7.6): an escalated phrase is coded again from the best of
    the coder's codes and the retrieved ones, by the table's retrieval stage, with the graph's
    vectors file. Each prediction records the layers that escalated it.

    `-O llm=on` adds L3 (task 8.3) with the settings' provider and model (`TEXT2ICD_LLM_*`), or
    `-O provider=... -O model=... -O effort=... -O budget=...`; `-O cache=off` skips the decision
    cache. Each prediction records its LLM calls, cost and token usage. `-O llm=batch` answers
    the decisions through Message Batches first (task 8.8, Anthropic API only, half price), then
    codes every phrase from the cache."""

    name = "coder"
    LLM_OPTIONS = ("llm", "provider", "model", "effort", "budget", "cache")

    def __init__(self, graph: Graph, settings: Settings, options: Options) -> None:
        self.width = option(options, "width", WIDTH)
        setting = options.get("setting", "inpatient")
        episode = options.get("episode", settings.default_episode)
        retrieval = options.get("retrieval", "off")
        unknown = set(options) - {"width", "setting", "episode", "retrieval", *self.LLM_OPTIONS}
        if unknown:
            raise SystemSetupError(f"unknown options for coder: {', '.join(sorted(unknown))}")
        if setting not in ("inpatient", "outpatient"):
            raise SystemSetupError(
                f"option setting must be inpatient or outpatient, got {setting!r}"
            )
        if episode not in EPISODES:
            raise SystemSetupError(f"option episode must be one of {', '.join(EPISODES)}")
        if retrieval not in ("on", "off"):
            raise SystemSetupError(f"option retrieval must be on or off, got {retrieval!r}")
        self.setting: Setting = "outpatient" if setting == "outpatient" else "inpatient"
        self.episode: Episode = next(e for e in EPISODES if e == episode)
        self.calibration = read_calibration(settings.calibration)
        self.batch: BatchModel | None = None
        self.navigator = self._navigator(settings, options)
        try:
            # No result cache: every phrase is coded and timed.
            self.resolver = Resolver(
                settings=settings,
                graph=graph,
                retrieval=retrieval == "on",
                navigator=self.navigator,
                calibration=self.calibration,
                width=self.width,
                cache_size=0,
            )
        except (ResolverError, CalibrationError) as exc:
            raise SystemSetupError(str(exc)) from exc
        self.coder = self.resolver.coder(self.episode)
        self.uses_llm = self.endpoint is not None

    def _navigator(self, settings: Settings, options: Options) -> Navigator | None:
        llm = options.get("llm", "off")
        cache = options.get("cache", "on")
        if llm not in ("on", "off", "batch"):
            raise SystemSetupError(f"option llm must be on, off or batch, got {llm!r}")
        if cache not in ("on", "off"):
            raise SystemSetupError(f"option cache must be on or off, got {cache!r}")
        if llm == "batch" and cache == "off":
            raise SystemSetupError("llm=batch keeps the batch answers in the decision cache")
        if llm == "off":
            given = sorted(set(options) & set(self.LLM_OPTIONS) - {"llm"})
            if given:
                raise SystemSetupError(f"options {', '.join(given)} need llm=on")
            return None
        if self.calibration is None:
            raise SystemSetupError("llm=on needs a calibration table: L3 runs on escalated phrases")
        try:
            budget = float(options["budget"]) if "budget" in options else None
        except ValueError:
            raise SystemSetupError("option budget must be a number of US dollars") from None
        named = {key: options.get(key) for key in ("provider", "model", "effort")}
        try:
            navigator = build_navigator(settings, budget=budget, cache=cache == "on", **named)
        except (LLMUnavailableError, PriceError) as exc:
            raise SystemSetupError(f"llm={llm}: {exc}") from exc
        if llm == "batch":
            if not isinstance(navigator, LLMNavigator) or navigator.cache is None:
                raise SystemSetupError("llm=batch needs a provider and a model")
            chooser = navigator.chooser
            if not isinstance(chooser, MessagesModel):
                raise SystemSetupError("llm=batch needs a Messages API model")
            meter = CostMeter(navigator.meter.price, navigator.meter.budget_usd, batch=True)
            try:
                self.batch = BatchModel(chooser, meter, navigator.cache, navigator.space)
            except BatchError as exc:
                raise SystemSetupError(str(exc)) from exc
        return navigator

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        navigator = self.navigator
        if self.batch is not None and isinstance(navigator, LLMNavigator):
            cache = self.batch.cache
            named = (navigator.provider, navigator.model)
            collector = Collector(cache, navigator.space, *named)
            self.coder.navigator = collector
            answered = answer_in_rounds(
                self._code, phrases, collector, self.batch, navigator.calls_per_phrase
            )
            self.coder.navigator = Replay(
                cache, navigator.space, *named, answered, navigator.calls_per_phrase
            )
        for phrase in phrases:
            yield self.predict(phrase)

    def _code(self, phrase: str) -> CodeSet:
        return self.coder.code(phrase, self.setting)

    def predict(self, phrase: str) -> Prediction:
        code_set = self._code(phrase)
        first = code_set.diagnoses[0].alternates if code_set.diagnoses else ()
        return Prediction(
            id="",
            codes=code_set.codes,
            alternates=first,
            confidence=code_set.confidence,
            escalated=code_set.escalated if self.calibration is not None else None,
            llm_calls=code_set.llm_calls,
            cost_usd=code_set.cost_usd,
            usage=code_set.usage,
        )

    def settings(self) -> dict[str, JsonValue]:
        fitted = self.calibration.fitted if self.calibration is not None else {}
        retriever = self.coder.retriever
        llm: dict[str, JsonValue] | None = None
        if isinstance(self.navigator, LLMNavigator):
            llm = {
                "provider": self.navigator.provider,
                "model": self.navigator.model,
                "effort": getattr(self.navigator.chooser, "effort", None),
                "calls_per_phrase": self.navigator.calls_per_phrase,
                "budget_usd": self.navigator.meter.budget_usd,
                "prompt_version": PROMPT_VERSION,
                "decision_cache": self.navigator.cache is not None,
                "batch": self.batch is not None,
            }
            if self.batch is not None:
                llm["budget_usd"] = self.batch.meter.budget_usd
        return {
            "width": self.width,
            "setting": self.setting,
            "episode": self.episode,
            "calibration": fitted.get("name"),
            "retrieval": retriever.name if retriever is not None else None,
            "llm": llm,
        }

    @property
    def endpoint(self) -> str | None:
        if self.navigator is None or self.navigator.provider == "none":
            return None
        return self.navigator.provider

    def warnings(self) -> list[str]:
        if self.batch is not None:
            warning = self.batch.meter.cache_warning()
        elif isinstance(self.navigator, LLMNavigator):
            warning = self.navigator.meter.cache_warning()
        else:
            return []
        return [warning] if warning else []

    def close(self) -> None:
        if self.navigator is not None:
            self.navigator.close()


Factory = Callable[[Graph, Settings, Options], System]

SYSTEMS: dict[str, Factory] = {
    "bm25": lambda graph, settings, options: BM25System(graph, options),
    "embedding": lambda graph, settings, options: EmbeddingSystem(graph, options),
    "llm": lambda graph, settings, options: _llm(graph, settings, options),
    "index": lambda graph, settings, options: IndexSystem(graph, options),
    "coder": lambda graph, settings, options: CoderSystem(graph, settings, options),
}


def _llm(graph: Graph, settings: Settings, options: Options) -> System:
    from text2icd.eval.llm_baseline import LLMChoiceSystem  # imports this module

    return LLMChoiceSystem(graph, settings, options)


def build_system(name: str, graph: Graph, settings: Settings, options: Options) -> System:
    try:
        factory = SYSTEMS[name]
    except KeyError:
        known = ", ".join(sorted(SYSTEMS))
        raise SystemSetupError(f"unknown system {name!r}; known systems: {known}") from None
    return factory(graph, settings, options)
