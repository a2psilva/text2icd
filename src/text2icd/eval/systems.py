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
from text2icd.resolve.coder import Coder
from text2icd.resolve.confidence import CalibrationError, read_calibration
from text2icd.resolve.retrieve import VectorsError, load_vectors
from text2icd.resolve.tabular import EPISODES, TabularVerifier
from text2icd.resolve.walk import WIDTH, IndexWalker

Options = Mapping[str, str]


class SystemSetupError(ValueError):
    """A system that cannot be built with the given options."""


class System(ABC):
    name: ClassVar[str]
    uses_llm: ClassVar[bool] = False

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
    vectors file. Each prediction records the layers that escalated it."""

    name = "coder"

    def __init__(self, graph: Graph, settings: Settings, options: Options) -> None:
        self.width = option(options, "width", WIDTH)
        setting = options.get("setting", "inpatient")
        episode = options.get("episode", settings.default_episode)
        retrieval = options.get("retrieval", "off")
        unknown = set(options) - {"width", "setting", "episode", "retrieval"}
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
        self.setting, self.episode = setting, episode
        walker = IndexWalker.from_graph(
            graph, "outpatient" if setting == "outpatient" else "inpatient", self.width
        )
        default = next(e for e in EPISODES if e == episode)
        self.calibration = read_calibration(settings.calibration)
        verifier = TabularVerifier(graph, walker.finder.forms, default)
        retriever = None
        if retrieval == "on":
            if self.calibration is None or self.calibration.retrieval is None:
                raise SystemSetupError(
                    "retrieval=on needs a calibration table with a retrieval stage "
                    "(text2icd eval calibrate)"
                )
            try:
                retriever = load_vectors(graph)
            except (VectorsError, ValueError) as exc:
                raise SystemSetupError(str(exc)) from exc
        try:
            self.coder = Coder(graph, walker, verifier, self.calibration, retriever)
        except CalibrationError as exc:
            raise SystemSetupError(str(exc)) from exc

    def predict(self, phrase: str) -> Prediction:
        code_set = self.coder.code(phrase)
        first = code_set.diagnoses[0].alternates if code_set.diagnoses else ()
        return Prediction(
            id="",
            codes=code_set.codes,
            alternates=first,
            confidence=code_set.confidence,
            escalated=code_set.escalated if self.calibration is not None else None,
        )

    def settings(self) -> dict[str, JsonValue]:
        fitted = self.calibration.fitted if self.calibration is not None else {}
        retriever = self.coder.retriever
        return {
            "width": self.width,
            "setting": self.setting,
            "episode": self.episode,
            "calibration": fitted.get("name"),
            "retrieval": retriever.name if retriever is not None else None,
        }


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
