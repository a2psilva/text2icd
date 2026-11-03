import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from text2icd.embeddings import Vectors, load_embedder, normalize
from text2icd.graph.model import Graph
from text2icd.resolve.retrieve import (
    EmbeddingRetriever,
    LexicalRetriever,
    Text,
    distinct,
    fuse,
    texts,
)


class Words:
    """A stand-in embedder: hashed word counts, so texts sharing words are similar."""

    model = "test/words"

    def encode(self, batch: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(batch), 64), dtype=np.float32)
        for row, text in enumerate(batch):
            for word in text.casefold().replace(",", " ").split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return normalize(vectors)


def test_texts(graph: Graph) -> None:
    tabular = texts(graph)
    assert {graph.code(t.node).kind for t in tabular} == {"diag"}  # no chapter or section
    assert (
        Text("E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease", "tabular")
        in tabular
    )
    assert len({(t.node, t.text.casefold()) for t in tabular}) == len(tabular)
    index = texts(graph, ("index",))
    assert Text("J69.0", "Pneumonia, aspiration", "index") in index
    assert {t.source for t in index} == {"index"}
    both = texts(graph, ("tabular", "index"))
    assert len(both) == len(tabular) + len(index)


def test_distinct_and_fuse() -> None:
    hits = [("A", 0.9), ("B", 0.8), ("A", 0.7), ("C", 0.6)]
    assert distinct(hits, 2) == [("A", 0.9), ("B", 0.8)]
    assert distinct(hits, 10) == [("A", 0.9), ("B", 0.8), ("C", 0.6)]
    fused = fuse([["A", "B", "A"], ["B", "C"]], 3)
    assert [node for node, _ in fused] == ["B", "A", "C"]  # B ranks in both
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fuse([["A", "B"]], 1) == [("A", pytest.approx(1 / 61))]
    assert fuse([], 5) == []


def test_embedding_retriever(graph: Graph) -> None:
    entries = texts(graph, ("tabular", "index"))
    retriever = EmbeddingRetriever(Words(), entries)
    assert retriever.name == "test/words"
    aspiration, diabetes = retriever.search(
        ["aspiration pneumonia", "type 2 diabetes mellitus with chronic kidney disease"], 3
    )
    assert aspiration[0][0] == "J69.0"
    assert len({node for node, _ in aspiration}) == len(aspiration) <= 3
    assert diabetes[0][0] == "E11.22"
    vectors = Words().encode([entry.text for entry in entries])
    same = EmbeddingRetriever(Words(), entries, vectors)
    assert same.search(["aspiration pneumonia"], 3) == [aspiration]
    with pytest.raises(ValueError, match="vectors for"):
        EmbeddingRetriever(Words(), entries, vectors[:2])


def test_lexical_retriever(graph: Graph) -> None:
    tabular = LexicalRetriever(graph)
    (found,) = tabular.search(["type 2 diabetes mellitus with diabetic chronic kidney disease"], 5)
    assert found[0][0] == "E11.22"
    assert all(graph.code(node).kind == "diag" for node, _ in found)
    both = LexicalRetriever(graph, ("tabular", "index"))
    query = ["aspiration pneumonia due to vomitus"]
    (pneumonia,) = both.search(query, 5)
    assert pneumonia[1][0] == "J69.0"  # first of the Index list, tied with E08 ("due to")
    assert "J69.0" not in {node for node, _ in tabular.search(query, 5)[0]}  # only the Index
    assert tabular.search(["zzzz"], 5) == [[]]


@pytest.mark.parametrize("spec", ["sentence:model", "onnx:", "model2vec", ""])
def test_load_embedder_refuses_unknown_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="expected model2vec"):
        load_embedder(spec)
