import hashlib
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from text2icd.config import Settings
from text2icd.embeddings import EmbeddingsUnavailableError, VectorIndex, Vectors, normalize
from text2icd.eval.systems import EmbeddingSystem, SystemSetupError, build_system
from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load


class BagOfWords:
    """A stand-in embedder: hashed word counts, so texts sharing words are similar. Words it did
    not see in its first call (the index) are ignored, as a real model ignores unknown tokens."""

    model = "test/bag-of-words"

    def __init__(self) -> None:
        self.calls = 0
        self.known: set[str] | None = None

    def encode(self, texts: Sequence[str]) -> Vectors:
        self.calls += 1
        words = [text.casefold().split() for text in texts]
        if self.known is None:
            self.known = {word for text in words for word in text}
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(words):
            for word in text:
                if word in self.known:
                    vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return normalize(vectors)


def test_normalize_and_search() -> None:
    vectors = normalize([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(vectors[0], [0.6, 0.8])
    assert np.allclose(vectors[1], [0.0, 0.0])
    index = VectorIndex(vectors)
    assert len(index) == 3
    (hits,) = index.search(normalize([[1.0, 0.1]]), 2)
    assert [row for row, _ in hits] == [2, 0]
    assert hits[0][1] == pytest.approx(0.995, abs=1e-3)
    assert index.search(normalize([[1.0, 0.0]]), 10)[0][0][0] == 2
    assert VectorIndex(np.zeros((0, 2), dtype=np.float32)).search(normalize([[1.0, 0.0]]), 3) == [
        []
    ]


def test_search_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import text2icd.embeddings as embeddings

    rng = np.random.default_rng(0)
    index = VectorIndex(normalize(rng.normal(size=(50, 8))))
    queries = normalize(rng.normal(size=(7, 8)))
    whole = index.search(queries, 5)
    monkeypatch.setattr(embeddings, "_SEARCH_CELLS", 60)
    chunked = index.search(queries, 5)
    # Smaller matrix products may round the last digit differently; the rows are the same.
    assert [[row for row, _ in hits] for hits in chunked] == [[r for r, _ in h] for h in whole]
    for hits, expected in zip(chunked, whole, strict=True):
        assert [score for _, score in hits] == pytest.approx([score for _, score in expected])


def test_embedding_system(graph: Graph) -> None:
    embedder = BagOfWords()
    system = EmbeddingSystem(graph, {"top": "3"}, embedder=embedder)
    phrases = ["type 2 diabetes mellitus with diabetic chronic kidney disease", "zzzz qqqq"]
    first, nothing = system.predict_many(phrases)
    assert first.primary == "E11.22"
    assert len(first.ranked) <= 3
    assert all(graph.code(code).billable for code in first.ranked)
    assert 0 < (first.confidence or 0) <= 1
    assert first.latency_ms > 0
    assert nothing.codes == ()
    assert system.predict(phrases[0]).primary == "E11.22"
    assert system.settings() == {"model": "test/bag-of-words", "k": 50, "top": 3}
    assert embedder.calls == 3  # the index once, then one batch per call


def test_embedding_options(graph: Graph) -> None:
    with pytest.raises(SystemSetupError, match="unknown options for embedding: kk"):
        EmbeddingSystem(graph, {"kk": "1"}, embedder=BagOfWords())


def test_missing_dependency(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def without_model2vec(name: str, *args: object, **kwargs: object) -> object:
        if name == "model2vec":
            raise ImportError("no model2vec")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", without_model2vec)
    with pytest.raises(EmbeddingsUnavailableError, match="uv sync --extra embeddings"):
        build_system("embedding", graph, Settings(), {})


@pytest.mark.slow
def test_release_embedding_baseline(release_graph: Callable[[str], BuildReport]) -> None:
    pytest.importorskip("model2vec")
    with load(path=release_graph("fy2027").path) as graph:
        system = build_system("embedding", graph, Settings(), {})
        diabetes, pneumonia = system.predict_many(
            [
                "type 2 diabetes mellitus with diabetic chronic kidney disease",
                "aspiration pneumonia",
            ]
        )
    assert diabetes.primary == "E11.22"
    assert "J69.0" in pneumonia.ranked
