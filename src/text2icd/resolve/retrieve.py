"""L2, local retrieval (Phase 7): the Tabular codes whose texts are nearest a phrase, for the
phrases the deterministic coder is not confident about. No API call.

The texts are the graph's own:

- `tabular`: code descriptions, inclusion terms and `includes` notes;
- `index`: the full path of each term of the Alphabetic Index and of the External Cause Index that
  gives a code ("Pneumonia, aspiration" for J69.0). The Neoplasm Table and the Table of Drugs are
  left out: their codes depend on a column the path does not name.

A retriever ranks the texts against a phrase, by cosine similarity of local embeddings
(`EmbeddingRetriever`) or by BM25 over the graph's full-text tables (`LexicalRetriever`), and
returns distinct codes, best first. Many are categories or partial codes: Tabular verification
completes them to billable codes, as it does the Index walk's candidates.
"""

from collections.abc import Iterable, Sequence
from typing import Literal, NamedTuple, Protocol

from text2icd.embeddings import Embedder, VectorIndex, Vectors
from text2icd.graph.build import PATH_SEPARATOR
from text2icd.graph.model import Graph

Source = Literal["tabular", "index"]
SOURCES: tuple[Source, ...] = ("tabular", "index")
INDEX_BOOKS = ("index", "eindex")
CODE_KINDS = ("diag", "generated")  # chapters and sections are not codes
RRF = 60  # the constant of reciprocal rank fusion
_OVERFETCH = 4  # texts read per code wanted, since several texts name one code

Hits = list[tuple[str, float]]  # (Tabular node, score), best first


class Text(NamedTuple):
    node: str  # the Tabular node the text names or the term gives
    text: str
    source: Source


def texts(graph: Graph, sources: Sequence[Source] = ("tabular",)) -> list[Text]:
    """The texts retrieval ranks, without repeats of one text for one code."""
    out: list[Text] = []
    seen: set[tuple[str, str]] = set()

    def add(node: str, text: str, source: Source) -> None:
        key = (node, text.casefold())
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
    """Cosine similarity between the phrase and every text, in one embedding space."""

    def __init__(
        self, embedder: Embedder, entries: Sequence[Text], vectors: Vectors | None = None
    ) -> None:
        """`vectors`, one row per entry, when they were embedded before (task 7.5)."""
        self.name = embedder.model
        self.embedder = embedder
        self.nodes = [entry.node for entry in entries]
        if vectors is None:
            vectors = embedder.encode([entry.text for entry in entries])
        if len(vectors) != len(entries):
            raise ValueError(f"{len(vectors)} vectors for {len(entries)} texts")
        self.index = VectorIndex(vectors)

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
