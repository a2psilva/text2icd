"""Systems the evaluation runner can score: the baselines now, the text2icd resolver later.

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


Factory = Callable[[Graph, Settings, Options], System]

SYSTEMS: dict[str, Factory] = {
    "bm25": lambda graph, settings, options: BM25System(graph, options),
    "embedding": lambda graph, settings, options: EmbeddingSystem(graph, options),
}


def build_system(name: str, graph: Graph, settings: Settings, options: Options) -> System:
    try:
        factory = SYSTEMS[name]
    except KeyError:
        known = ", ".join(sorted(SYSTEMS))
        raise SystemSetupError(f"unknown system {name!r}; known systems: {known}") from None
    return factory(graph, settings, options)
