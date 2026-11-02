"""Local text embeddings: static models run with numpy only (no torch, no network at encode time
once the model is cached). Install with `uv sync --extra embeddings`.

Used by the B2 baseline now, and by local retrieval (L2) later. The model id is configuration and
is recorded with every result.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

# Rows of scores held at once when searching (queries x indexed texts).
_SEARCH_CELLS = 20_000_000


class EmbeddingsUnavailableError(RuntimeError):
    """The optional embedding dependencies are not installed."""


class Embedder(Protocol):
    model: str

    def encode(self, texts: Sequence[str]) -> Vectors:
        """One L2-normalised row per text."""
        ...


def normalize(vectors: npt.ArrayLike) -> Vectors:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    return array / norms


class StaticEmbedder:
    """A model2vec static model from the Hugging Face Hub (downloaded once, then cached)."""

    def __init__(self, model: str) -> None:
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise EmbeddingsUnavailableError(
                "local embeddings need the optional dependencies: uv sync --extra embeddings"
            ) from exc
        self.model = model
        self._model = StaticModel.from_pretrained(model)

    def encode(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._model.dim), dtype=np.float32)
        return normalize(self._model.encode(list(texts), show_progress_bar=False))


class VectorIndex:
    """Normalised vectors searched by cosine similarity."""

    def __init__(self, vectors: Vectors) -> None:
        self.vectors = normalize(vectors)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, queries: Vectors, k: int) -> list[list[tuple[int, float]]]:
        """For each query, the `k` most similar rows as (row, cosine), best first."""
        k = min(k, len(self))
        if k == 0:
            return [[] for _ in range(len(queries))]
        chunk = max(1, _SEARCH_CELLS // max(len(self), 1))
        results: list[list[tuple[int, float]]] = []
        for start in range(0, len(queries), chunk):
            scores = queries[start : start + chunk] @ self.vectors.T
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            for row, candidates in zip(scores, top, strict=True):
                ordered = candidates[np.argsort(-row[candidates], kind="stable")]
                results.append([(int(i), float(row[i])) for i in ordered])
        return results
