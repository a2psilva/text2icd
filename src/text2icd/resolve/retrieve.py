"""L2, local retrieval (Phase 7): the Tabular codes whose texts are nearest a phrase, for the
phrases the deterministic coder is not confident about. No API call.

The texts are the graph's own:

- `tabular`: code descriptions, inclusion terms and `includes` notes;
- `index`: the full path of each term of the Alphabetic Index and of the External Cause Index that
  gives a code ("Pneumonia, aspiration" for J69.0). The Neoplasm Table and the Table of Drugs are
  left out: their codes depend on a column the path does not name.

Texts with the same words for the same code ("Pneumonia, aspiration" and "Aspiration pneumonia")
are kept once.

A retriever ranks the texts against a phrase, by cosine similarity of local embeddings
(`EmbeddingRetriever`) or by BM25 over the graph's full-text tables (`LexicalRetriever`), and
returns distinct codes, best first. Many are categories or partial codes: Tabular verification
completes them to billable codes, as it does the Index walk's candidates.

The embeddings of the texts are built once per graph (`build_vectors`, `text2icd graph embed` or
`text2icd build-graph --embeddings`) into a float16 matrix next to the SQLite file, with a JSON
file naming the model, the texts' sources, the graph it belongs to and the code of each row
(task 7.5). `load_vectors` refuses vectors built for another graph.
"""

import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, ValidationError

from text2icd import __version__
from text2icd.embeddings import Embedder, VectorIndex, load_embedder, normalize
from text2icd.graph.build import PATH_SEPARATOR
from text2icd.graph.model import Graph

Source = Literal["tabular", "index"]
SOURCES: tuple[Source, ...] = ("tabular", "index")
INDEX_BOOKS = ("index", "eindex")
CODE_KINDS = ("diag", "generated")  # chapters and sections are not codes
RRF = 60  # the constant of reciprocal rank fusion
_OVERFETCH = 4  # texts read per code wanted, since several texts name one code
_WORD = re.compile(r"\w+")

# L2's embedding model and texts, chosen by the benchmark of task 7.4 (docs/evaluation.md). The
# full-precision weights, whose vectors do not depend on the batch (`embeddings.OnnxEmbedder`).
DEFAULT_MODEL = "onnx:Xenova/bge-small-en-v1.5:onnx/model.onnx"
DEFAULT_SOURCES: tuple[Source, ...] = ("tabular", "index")
MAX_BYTES = 100_000_000  # the budget of the vectors file
VECTORS_FORMAT = 1
_ENCODE_BATCH = 8192  # texts embedded between progress reports

Hits = list[tuple[str, float]]  # (Tabular node, score), best first


class Text(NamedTuple):
    node: str  # the Tabular node the text names or the term gives
    text: str
    source: Source


def texts(graph: Graph, sources: Sequence[Source] = ("tabular",)) -> list[Text]:
    """The texts retrieval ranks, each set of words once per code."""
    out: list[Text] = []
    seen: set[tuple[str, frozenset[str]]] = set()

    def add(node: str, text: str, source: Source) -> None:
        key = (node, frozenset(_WORD.findall(text.casefold())))
        if key not in seen:
            seen.add(key)
            out.append(Text(node, text, source))

    if "tabular" in sources:
        for row in graph.tabular_texts():
            if graph.code(row.node).kind in CODE_KINDS:
                add(row.node, row.text, "tabular")
    if "index" in sources:
        for node, path in _index_paths(graph):
            add(node, path, "index")
    return out


def _index_paths(graph: Graph) -> Iterable[tuple[str, str]]:
    for term in graph.terms():
        if term.book not in INDEX_BOOKS:
            continue
        for pointer in graph.pointers(term.id):
            if pointer.field == "code" and pointer.target is not None:
                yield pointer.target, term.path.replace(PATH_SEPARATOR, ", ")


class Retriever(Protocol):
    name: str

    def search(self, queries: Sequence[str], k: int) -> list[Hits]:
        """For each query, up to `k` distinct Tabular nodes, best first."""
        ...


def distinct(hits: Iterable[tuple[str, float]], k: int) -> Hits:
    """The first hit of each node, up to `k` nodes."""
    out: dict[str, float] = {}
    for node, score in hits:
        if node not in out:
            out[node] = score
            if len(out) == k:
                break
    return list(out.items())


class EmbeddingRetriever:
    """Cosine similarity between the phrase and every text, in one embedding space: `nodes[i]`
    is the code of the text embedded in row `i` of `vectors`."""

    def __init__(
        self,
        embedder: Embedder,
        nodes: Sequence[str],
        vectors: npt.NDArray[np.floating],
        sources: Sequence[Source] = (),
    ) -> None:
        """`sources`, where the texts come from, when known."""
        if len(vectors) != len(nodes):
            raise ValueError(f"{len(vectors)} vectors for {len(nodes)} texts")
        self.name = embedder.model
        self.embedder = embedder
        self.nodes = list(nodes)
        self.sources = tuple(sources)
        self.index = VectorIndex(vectors)

    @classmethod
    def embed(cls, embedder: Embedder, entries: Sequence[Text]) -> "EmbeddingRetriever":
        """Embed the texts now (the benchmark); L2 reads them from the graph's vectors file."""
        vectors = embedder.encode([entry.text for entry in entries])
        sources = tuple(dict.fromkeys(entry.source for entry in entries))
        return cls(embedder, [entry.node for entry in entries], vectors, sources)

    def search(self, queries: Sequence[str], k: int) -> list[Hits]:
        found = self.index.search(self.embedder.encode(list(queries)), k * _OVERFETCH)
        return [distinct(((self.nodes[row], score) for row, score in hits), k) for hits in found]


class LexicalRetriever:
    """BM25 over the graph's full-text tables: Tabular texts, Index terms, or both fused by
    reciprocal rank."""

    name = "bm25"

    def __init__(self, graph: Graph, sources: Sequence[Source] = ("tabular",)) -> None:
        self.graph = graph
        self.sources = tuple(sources)
        self._targets: dict[int, list[str]] = {}
        if "index" in self.sources:
            for term in graph.terms():
                if term.book in INDEX_BOOKS:
                    targets = [
                        p.target
                        for p in graph.pointers(term.id)
                        if p.field == "code" and p.target is not None
                    ]
                    if targets:
                        self._targets[term.id] = targets

    def search(self, queries: Sequence[str], k: int) -> list[Hits]:
        return [self._one(query, k) for query in queries]

    def _one(self, query: str, k: int) -> Hits:
        ranked: list[list[str]] = []
        wanted = k * _OVERFETCH
        if "tabular" in self.sources:
            rows = self.graph.search_tabular(query, wanted)
            ranked.append(
                [row.node for row in rows if self.graph.code(row.node).kind in CODE_KINDS]
            )
        if "index" in self.sources:
            terms = self.graph.search_index(query, wanted)
            ranked.append([node for hit in terms for node in self._targets.get(hit.term, ())])
        return fuse(ranked, k)


def fuse(rankings: Sequence[Sequence[str]], k: int) -> Hits:
    """Reciprocal rank fusion of ranked node lists (each node at its first rank in each list)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (node, _) in enumerate(distinct(((n, 0.0) for n in ranking), len(ranking))):
            scores[node] = scores.get(node, 0.0) + 1 / (RRF + rank + 1)
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return ordered[:k]


# -- The vectors file (task 7.5) -------------------------------------------------------------------


class VectorsError(Exception):
    """The vectors file is missing, malformed, or built for another graph or model."""


class VectorsMeta(BaseModel):
    """What the vectors file holds, in the JSON file next to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: int = VECTORS_FORMAT
    model: str  # as `embeddings.load_embedder` reads it
    model_id: str  # the embedder's own name, with its weights file
    revision: str | None = None  # the model's Hub commit; None in files from before 0.2
    sources: tuple[Source, ...]
    release: str
    graph: str  # the graph's content_sha256
    rows: int
    dim: int
    text2icd: str = __version__
    nodes: tuple[str, ...]  # the code of each row


@dataclass(frozen=True, slots=True)
class BuiltVectors:
    path: Path
    meta: VectorsMeta
    size: int  # bytes of the matrix file
    seconds: float

    def lines(self) -> list[str]:
        budget = "" if self.size <= MAX_BYTES else f", over the {MAX_BYTES / 1e6:.0f} MB budget"
        return [
            f"{self.path} ({self.size / 1e6:.1f} MB{budget})",
            f"  {self.meta.rows:,} texts ({'+'.join(self.meta.sources)}) x {self.meta.dim} "
            f"dimensions, {self.meta.model}, {self.seconds:.0f} s",
        ]


def vectors_path(graph_path: Path) -> Path:
    """`text2icd-graph-fy2027.vectors.npy` next to `text2icd-graph-fy2027.sqlite`."""
    return graph_path.with_suffix(".vectors.npy")


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def build_vectors(
    graph: Graph,
    embedder: Embedder,
    model: str,
    path: Path | None = None,
    sources: Sequence[Source] = DEFAULT_SOURCES,
    progress: Callable[[int], None] | None = None,
) -> BuiltVectors:
    """Embed the graph's texts with `embedder` (named `model`, as `load_embedder` reads it) and
    write the float16 matrix and its JSON file, by default next to the graph."""
    path = path or vectors_path(graph.path)
    entries = texts(graph, sources)
    began = time.perf_counter()
    blocks = []
    for start in range(0, len(entries), _ENCODE_BATCH):
        batch = entries[start : start + _ENCODE_BATCH]
        blocks.append(normalize(embedder.encode([entry.text for entry in batch])))
        if progress is not None:
            progress(len(batch))
    vectors = np.concatenate(blocks).astype(np.float16) if blocks else np.zeros((0, 0), np.float16)
    seconds = time.perf_counter() - began
    meta = VectorsMeta(
        model=model,
        model_id=embedder.model,
        revision=getattr(embedder, "revision", None),
        sources=tuple(sources),
        release=graph.release,
        graph=graph.meta.get("content_sha256", ""),
        rows=len(entries),
        dim=int(vectors.shape[1]),
        nodes=tuple(entry.node for entry in entries),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as fh:
        np.save(fh, vectors, allow_pickle=False)
    partial.replace(path)
    _meta_path(path).write_text(json.dumps(meta.model_dump(mode="json")) + "\n", "utf-8")
    return BuiltVectors(path, meta, path.stat().st_size, seconds)


def load_vectors(
    graph: Graph, path: Path | None = None, embedder: Embedder | None = None
) -> EmbeddingRetriever:
    """The retriever over the graph's vectors file, with the model it was built with, at the
    same Hub revision (or `embedder`, of the same dimension). The matrix is memory-mapped: it
    costs no load time and no memory until it is searched, and processes share it through the
    page cache."""
    path, meta = vectors_meta(graph, path)
    rebuild = f"run `text2icd graph embed -r {graph.release}`"
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.dtype != np.float16 or vectors.shape != (meta.rows, meta.dim):
        raise VectorsError(f"{path} does not match its .json file; {rebuild}")
    embedder = embedder or load_embedder(meta.model, meta.revision)
    dim = getattr(embedder, "dim", meta.dim)
    if dim != meta.dim:
        raise VectorsError(f"{path} has {meta.dim} dimensions, the model {dim}; {rebuild}")
    return EmbeddingRetriever(embedder, meta.nodes, vectors, meta.sources)


def vectors_meta(graph: Graph, path: Path | None = None) -> tuple[Path, VectorsMeta]:
    """The vectors file of the graph and what its JSON file says, checked against the graph;
    VectorsError when either file is missing or was built for another graph."""
    path = path or vectors_path(graph.path)
    rebuild = f"run `text2icd graph embed -r {graph.release}`"
    meta = read_vectors_meta(path, rebuild)
    if meta.format != VECTORS_FORMAT:
        raise VectorsError(f"{path} has format {meta.format}, this text2icd reads {VECTORS_FORMAT}")
    if meta.graph != graph.meta.get("content_sha256"):
        raise VectorsError(f"{path} was built for another graph; {rebuild}")
    return path, meta


def read_vectors_meta(path: Path, rebuild: str = "run `text2icd graph embed`") -> VectorsMeta:
    """What the JSON file of the vectors file at `path` says, unchecked against a graph."""
    if not path.is_file() or not _meta_path(path).is_file():
        raise VectorsError(f"{path} or its .json file does not exist; {rebuild}")
    try:
        return VectorsMeta.model_validate_json(_meta_path(path).read_text("utf-8"))
    except ValidationError as exc:
        raise VectorsError(f"{_meta_path(path)}: {exc}; {rebuild}") from exc


class LazyRetriever:
    """L2's retriever, loaded on its first search (task 9.2). Most phrases are never escalated,
    so a resolver starts without reading the vectors or the embedding model. Its `name`, which
    the calibration's retrieval stage is checked against, comes from the vectors' JSON file."""

    def __init__(self, graph: Graph, path: Path | None = None) -> None:
        self._graph = graph
        self._path, meta = vectors_meta(graph, path)
        self.name = meta.model_id
        self.model, self.revision = meta.model, meta.revision
        self._loaded: EmbeddingRetriever | None = None
        self._loading = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> EmbeddingRetriever:
        with self._loading:
            if self._loaded is None:
                self._loaded = load_vectors(self._graph, self._path)
        return self._loaded

    def search(self, queries: Sequence[str], k: int) -> list[Hits]:
        return self.load().search(queries, k)
