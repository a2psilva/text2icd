import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import text2icd.embeddings as embeddings
from text2icd.embeddings import VectorIndex, Vectors, load_embedder, normalize
from text2icd.graph.model import Graph, TabularText
from text2icd.resolve.retrieve import (
    MAX_BYTES,
    EmbeddingRetriever,
    LexicalRetriever,
    Text,
    VectorsError,
    build_vectors,
    distinct,
    fuse,
    load_vectors,
    texts,
    vectors_path,
)


class Words:
    """A stand-in embedder: hashed word counts, so texts sharing words are similar."""

    model = "test/words"
    dim = 64

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
    assert len({(t.node, frozenset(t.text.casefold().split())) for t in tabular}) == len(tabular)
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
    retriever = EmbeddingRetriever.embed(Words(), entries)
    assert retriever.name == "test/words"
    aspiration, diabetes = retriever.search(
        ["aspiration pneumonia", "type 2 diabetes mellitus with chronic kidney disease"], 3
    )
    assert aspiration[0][0] == "J69.0"
    assert len({node for node, _ in aspiration}) == len(aspiration) <= 3
    assert diabetes[0][0] == "E11.22"
    vectors = Words().encode([entry.text for entry in entries])
    nodes = [entry.node for entry in entries]
    same = EmbeddingRetriever(Words(), nodes, vectors)
    assert same.search(["aspiration pneumonia"], 3) == [aspiration]
    with pytest.raises(ValueError, match="vectors for"):
        EmbeddingRetriever(Words(), nodes, vectors[:2])


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


def test_texts_keep_each_set_of_words_once_per_code() -> None:
    rows = [
        TabularText(1, "J69.0", "description", "Pneumonitis due to inhalation of food"),
        TabularText(2, "J69.0", "inclusion", "Aspiration pneumonia"),
        TabularText(3, "J69.1", "inclusion", "Aspiration pneumonia"),  # another code
    ]
    term = SimpleNamespace(id=7, book="index", path="Pneumonia > aspiration")
    pointer = SimpleNamespace(field="code", target="J69.0")
    graph = SimpleNamespace(
        tabular_texts=lambda: iter(rows),
        code=lambda key: SimpleNamespace(kind="diag"),
        terms=lambda: iter([term]),
        pointers=lambda term_id: (pointer,),
    )
    found = texts(cast(Graph, graph), ("tabular", "index"))
    assert [(t.node, t.source) for t in found] == [
        ("J69.0", "tabular"),
        ("J69.0", "tabular"),
        ("J69.1", "tabular"),
    ]  # "Pneumonia, aspiration" repeats the words of J69.0's inclusion term


def test_float16_vectors_are_searched_like_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    vectors = normalize(rng.normal(size=(300, 16)))
    queries = normalize(rng.normal(size=(5, 16)))
    wide = VectorIndex(vectors).search(queries, 10)
    monkeypatch.setattr(embeddings, "_WIDEN_ROWS", 7)
    half = VectorIndex(vectors.astype(np.float16))
    assert half.vectors.dtype == np.float16
    narrow = half.search(queries, 10)
    for a, b in zip(wide, narrow, strict=True):
        assert [score for _, score in a] == pytest.approx([score for _, score in b], abs=2e-3)
        assert len({row for row, _ in a} & {row for row, _ in b}) >= 8


def test_build_and_load_vectors(graph: Graph, tmp_path: Path) -> None:
    built = build_vectors(graph, Words(), "test:words")
    assert built.path == vectors_path(graph.path)
    assert built.path.name.endswith(".vectors.npy")
    entries = texts(graph, ("tabular", "index"))
    meta = built.meta
    assert (meta.rows, meta.dim, meta.model, meta.model_id) == (
        len(entries),
        64,
        "test:words",
        "test/words",
    )
    assert meta.sources == ("tabular", "index")
    assert meta.graph == graph.meta["content_sha256"]
    assert meta.nodes == tuple(entry.node for entry in entries)
    assert np.load(built.path).dtype == np.float16
    assert built.size < MAX_BYTES
    assert "texts (tabular+index) x 64 dimensions, test:words" in "\n".join(built.lines())
    loaded = load_vectors(graph, embedder=Words())
    assert loaded.index.vectors.dtype == np.float16
    fresh = EmbeddingRetriever.embed(Words(), entries)
    query = ["aspiration pneumonia", "type 2 diabetes mellitus with chronic kidney disease"]
    assert [h[0][0] for h in loaded.search(query, 3)] == [h[0][0] for h in fresh.search(query, 3)]
    elsewhere = build_vectors(graph, Words(), "test:words", tmp_path / "v.npy", ("tabular",))
    assert load_vectors(graph, elsewhere.path, Words()).nodes == [e.node for e in texts(graph)]


def _edit_meta(path: Path, **changes: Any) -> None:
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta.update(changes)
    meta_path.write_text(json.dumps(meta))


def test_load_vectors_refuses_mismatches(graph: Graph, tmp_path: Path) -> None:
    with pytest.raises(VectorsError, match="does not exist; run `text2icd graph embed"):
        load_vectors(graph, tmp_path / "missing.npy", Words())
    path = build_vectors(graph, Words(), "test:words", tmp_path / "v.npy").path

    class Narrow(Words):
        dim = 32

    with pytest.raises(VectorsError, match="64 dimensions, the model 32"):
        load_vectors(graph, path, Narrow())
    _edit_meta(path, rows=3)
    with pytest.raises(VectorsError, match=r"does not match its \.json file"):
        load_vectors(graph, path, Words())
    _edit_meta(path, graph="0" * 64)
    with pytest.raises(VectorsError, match="built for another graph"):
        load_vectors(graph, path, Words())
    _edit_meta(path, format=99)
    with pytest.raises(VectorsError, match="format 99"):
        load_vectors(graph, path, Words())
    _edit_meta(path, unexpected=1)
    with pytest.raises(VectorsError, match="unexpected"):
        load_vectors(graph, path, Words())


def test_load_embedder_reads_the_onnx_file(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(embeddings, "OnnxEmbedder", lambda *args: calls.append(args))
    load_embedder("onnx:Xenova/bge-small-en-v1.5")
    load_embedder("onnx:Xenova/bge-small-en-v1.5:onnx/model.onnx")
    assert calls == [("Xenova/bge-small-en-v1.5",), ("Xenova/bge-small-en-v1.5", "onnx/model.onnx")]
