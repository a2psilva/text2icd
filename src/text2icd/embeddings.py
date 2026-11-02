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


class OnnxEmbedder:
    """A transformer encoder exported to ONNX (such as SapBERT), run with onnxruntime on the CPU:
    the [CLS] vector of each text, L2-normalised. Downloaded once from the Hugging Face Hub."""

    def __init__(
        self,
        repo: str,
        file: str = "onnx/model_quantized.onnx",
        *,
        max_length: int = 64,
        batch_size: int = 256,
    ) -> None:
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingsUnavailableError(
                "local embeddings need the optional dependencies: uv sync --extra embeddings"
            ) from exc
        self.model = f"{repo}:{file}"
        self.batch_size = batch_size
        self._tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            hf_hub_download(repo, file), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self.dim = int(self._session.get_outputs()[0].shape[-1])

    def encode(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Texts of similar length batched together pad less.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self.batch_size):
            rows = order[start : start + self.batch_size]
            encoded = self._tokenizer.encode_batch([texts[i] for i in rows])
            feeds = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
            }
            hidden = self._session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})
            out[rows] = np.asarray(hidden[0])[:, 0, :]
        return normalize(out)
