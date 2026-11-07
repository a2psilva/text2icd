"""The resolver (tasks 9.2 and 9.3): codes phrases with one release's graph, end to end.

`Resolver` is the one place that builds the cascade from the settings:
- the graph;
- the Index walker, one for both settings, since the setting is given per phrase;
- the Tabular verifier, one per episode of care, built when first asked for;
- the calibration table;
- L2's retriever, loaded on the first escalated phrase (`retrieve.LazyRetriever`);
- L3's navigator, when asked for.

`code` returns the public `CodingResult`, and `code_set` the coder's own `CodeSet`, for
evaluation. A small in-process cache returns a phrase coded before with the same settings
without coding it again (task 9.3); its results are marked `cached`. It lives in memory only:
nothing derived from a phrase is written to disk.

Once built, the resolver's objects (about 1.2 million for a full release) are moved out of the
garbage collector's reach with `gc.freeze()`. A full collection over them takes about 0.2 s and
would otherwise land on whichever phrase is being coded; they are freed as usual, by reference
counting, when the resolver is dropped, since they hold no cycles.
"""

import gc
import itertools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Sequence
from types import TracebackType
from typing import Any, Literal, Self

from text2icd import __version__, logs
from text2icd.config import Settings
from text2icd.embeddings import downloaded, extra_for, missing
from text2icd.extras import install_hint
from text2icd.graph.model import Graph, load
from text2icd.llm.navigator import Navigator, NoNavigator, build_navigator
from text2icd.resolve.coder import Coder, CodeSet
from text2icd.resolve.confidence import Calibration, read_calibration
from text2icd.resolve.retrieve import LazyRetriever, VectorsError, vectors_meta, vectors_path
from text2icd.resolve.tabular import Episode, TabularVerifier
from text2icd.resolve.walk import WIDTH, IndexWalker
from text2icd.result import CodingResult
from text2icd.text.model import Setting

CACHE_SIZE = 4096  # code sets kept; 0 turns the cache off
BATCH = 256  # phrases `code_many` codes together
MAX_PHRASE = 1_000  # characters: longer input is refused, not coded
_SHIPPED: Literal["shipped"] = "shipped"


class ResolverError(ValueError):
    """The resolver cannot start as asked: L2 or L3 is unavailable, or the input is refused."""


class Resolver:
    """Codes phrases with one release's graph: `code(phrase) -> CodingResult`. Safe to share
    between threads."""

    def __init__(
        self,
        release: str | None = None,
        *,
        settings: Settings | None = None,
        graph: Graph | None = None,
        retrieval: bool | None = None,
        llm: bool = False,
        navigator: Navigator | None = None,
        calibration: Calibration | None = None,
        width: int = WIDTH,
        cache_size: int = CACHE_SIZE,
    ) -> None:
        """`release`, or `graph` already loaded (then the caller closes it). `retrieval`: L2 on
        (True), off (False) or when its vectors file, a retrieval stage and the `embeddings`
        extra are there (None). `llm`: L3 with the settings' provider and model, or `navigator`.
        `calibration`: a table other than the settings' or the shipped one."""
        self.settings = settings or Settings()
        self._owns_graph = graph is None
        self.graph = graph if graph is not None else load(release, settings=self.settings)
        try:
            self.calibration = (
                calibration
                if calibration is not None
                else read_calibration(self.settings.calibration)
            )
            self.walker = IndexWalker.from_graph(self.graph, width=width)
            self.warnings: list[str] = []
            self.retriever = self._retriever(retrieval)
            if navigator is None and llm:
                navigator = build_navigator(self.settings)
                if isinstance(navigator, NoNavigator):
                    raise ResolverError(
                        "L3 needs a provider and a model: set TEXT2ICD_LLM_PROVIDER and "
                        "TEXT2ICD_LLM_MODEL"
                    )
            self.navigator = None if isinstance(navigator, NoNavigator) else navigator
            self._coders: dict[Episode, Coder] = {}
            self._building = threading.Lock()
            self._cache = _Cache(cache_size)
            self.coder()  # the default episode's, so that a mismatched table fails now
            gc.freeze()
        except BaseException:
            self.close()
            raise

    # -- Coding -----------------------------------------------------------------------------

    def code(
        self,
        phrase: str,
        *,
        setting: Setting = "inpatient",
        episode: Episode | None = None,
        explain: bool = False,
    ) -> CodingResult:
        """The phrase's code set as the public result; `explain` adds the paths and decisions a
        UI draws (`CodingResult.detail`)."""
        began = time.perf_counter()
        code_set, cached = self._coded(phrase, setting, episode)
        self._log(phrase, code_set, cached, (time.perf_counter() - began) * 1000)
        return CodingResult.of(code_set, self.graph, explain=explain, cached=cached)

    def code_many(
        self,
        phrases: Iterable[str],
        *,
        setting: Setting = "inpatient",
        episode: Episode | None = None,
        explain: bool = False,
        batch_size: int = BATCH,
    ) -> Iterator[CodingResult]:
        """The result of each phrase, in order, for any number of phrases (task 9.5), streamed
        `batch_size` at a time. A phrase repeated within a batch is coded once, even with the
        cache off, and its repeats are marked `cached`.

        Each phrase is coded alone, so its result never depends on the phrases around it.
        Searching a batch's escalated phrases together was measured: about 24% faster on
        `mimic-silver` dev, but the encoder's batch padding moved L2's confidence in the fourth
        decimal for 0.2% of phrases."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {batch_size}")
        return self._stream(iter(phrases), setting, episode, explain, batch_size)

    def code_set(
        self, phrase: str, *, setting: Setting = "inpatient", episode: Episode | None = None
    ) -> CodeSet:
        """The coder's own code set for the phrase."""
        return self._coded(phrase, setting, episode)[0]

    def coder(self, episode: Episode | None = None) -> Coder:
        """The coder for one episode of care (the 7th character assumed when the phrase states
        none); they share everything but the Tabular verifier."""
        episode = episode or self.settings.default_episode
        found = self._coders.get(episode)
        if found is not None:
            return found
        with self._building:
            found = self._coders.get(episode)
            if found is None:
                verifier = TabularVerifier(self.graph, self.walker.finder.forms, episode)
                found = Coder(
                    self.graph,
                    self.walker,
                    verifier,
                    self.calibration,
                    self.retriever,
                    self.navigator,
                )
                self._coders[episode] = found
        return found

    def _coded(
        self, phrase: str, setting: Setting, episode: Episode | None
    ) -> tuple[CodeSet, bool]:
        _check(phrase)
        episode = episode or self.settings.default_episode
        key = (phrase, setting, episode)
        found = self._cache.get(key)
        if found is not None:
            return found, True
        code_set = self.coder(episode).code(phrase, setting)
        self._cache.put(key, code_set)
        return code_set, False

    def _stream(
        self,
        phrases: Iterator[str],
        setting: Setting,
        episode: Episode | None,
        explain: bool,
        batch_size: int,
    ) -> Iterator[CodingResult]:
        while batch := list(itertools.islice(phrases, batch_size)):
            yield from self._batch(batch, setting, episode, explain)

    def _batch(
        self, phrases: Sequence[str], setting: Setting, episode: Episode | None, explain: bool
    ) -> list[CodingResult]:
        began = time.perf_counter()
        done: dict[str, tuple[CodeSet, bool]] = {}
        for phrase in phrases:
            if phrase not in done:
                done[phrase] = self._coded(phrase, setting, episode)
        results = []
        seen: set[str] = set()
        for phrase in phrases:
            code_set, hit = done[phrase]
            repeat = hit or phrase in seen
            seen.add(phrase)
            self._log(phrase, code_set, repeat, None)
            results.append(CodingResult.of(code_set, self.graph, explain=explain, cached=repeat))
        logs.event(
            "batch",
            phrases=len(phrases),
            coded=sum(not hit for _, hit in done.values()),
            ms=round((time.perf_counter() - began) * 1000, 3),
        )
        return results

    def _log(self, phrase: str, code_set: CodeSet, cached: bool, ms: float | None) -> None:
        if not logs.LOGGER.isEnabledFor(logging.INFO):
            return
        fields = logs.phrase_fields(phrase, self.settings.log_inputs)
        fields.update(
            codes=len(code_set.sequence),
            resolver=code_set.resolver,
            escalated=list(code_set.escalated),
            cached=cached,
            llm_calls=0 if cached else code_set.llm_calls,
            cost_usd=0.0 if cached else code_set.cost_usd,
        )
        if ms is not None:
            fields["ms"] = round(ms, 3)
        if self.settings.log_inputs:
            fields["coded"] = list(code_set.codes)
        logs.event("coded", **fields)

    # -- What it runs ------------------------------------------------------------------------

    def warm(self) -> None:
        """Load what is otherwise loaded on first use (L2's vectors and model), so that a server
        answers its first escalated phrase as fast as the others."""
        if isinstance(self.retriever, LazyRetriever):
            self.retriever.load()
            gc.freeze()

    def info(self) -> dict[str, Any]:
        """What this resolver runs, for `/version` and run headers."""
        stage = self.calibration.retrieval if self.calibration is not None else None
        calibration = str(self.settings.calibration) if self.settings.calibration else _SHIPPED
        return {
            "text2icd": __version__,
            "release": self.graph.release,
            "graph": self.graph.meta.get("content_sha256", ""),
            "calibration": calibration if self.calibration is not None else None,
            "retrieval": self.retriever.name if self.retriever is not None else None,
            "retrieval_stage": stage.model if stage is not None else None,
            "llm": (
                {"provider": self.navigator.provider, "model": self.navigator.model}
                if self.navigator is not None and hasattr(self.navigator, "provider")
                else None
            ),
        }

    def _retriever(self, wanted: bool | None) -> LazyRetriever | None:
        if wanted is False:
            return None
        stage = self.calibration.retrieval if self.calibration is not None else None
        auto = wanted is None
        if stage is None:
            if auto:
                return None
            raise ResolverError(
                "L2 needs a calibration table with a retrieval stage (text2icd eval calibrate)"
            )
        path = vectors_path(self.graph.path)
        if auto and not path.is_file():
            return None
        try:
            _, meta = vectors_meta(self.graph, path)
        except VectorsError as exc:
            if auto:
                self.warnings.append(f"L2 is off: {exc}")
                return None
            raise ResolverError(str(exc)) from exc
        lacking = missing(meta.model)
        if lacking:
            message = f"L2 needs {', '.join(lacking)}: {install_hint(extra_for(meta.model))}"
            if auto:
                self.warnings.append(f"L2 is off: {message}")
                return None
            raise ResolverError(message)
        if auto and not downloaded(meta.model, meta.revision):
            # Downloading it would stall the first escalated phrase, and fail offline.
            self.warnings.append(
                f"L2 is off: its model ({meta.model_id}) is not downloaded yet; "
                "--retrieval downloads it on the first escalated phrase"
            )
            return None
        return LazyRetriever(self.graph, path)

    # -- Lifecycle ---------------------------------------------------------------------------

    def close(self) -> None:
        navigator = getattr(self, "navigator", None)
        if navigator is not None:
            navigator.close()
        if self._owns_graph and hasattr(self, "graph"):
            self.graph.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        layers = ["L1"]
        if self.retriever is not None:
            layers.append("L2")
        if self.navigator is not None:
            layers.append("L3")
        return f"Resolver({self.graph.release!r}, {'+'.join(layers)})"


def _check(phrase: str) -> None:
    if len(phrase) > MAX_PHRASE:
        size = f"{len(phrase):,} characters"
        raise ResolverError(f"the phrase has {size}; at most {MAX_PHRASE:,} are coded")


class _Cache:
    """A bounded least-recently-used map, safe between threads."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[Hashable, CodeSet] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> CodeSet | None:
        if not self.size:
            return None
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
            return found

    def put(self, key: Hashable, value: CodeSet) -> None:
        if not self.size:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
