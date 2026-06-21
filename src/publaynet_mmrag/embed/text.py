"""Text embedding with BGE-M3 (dense + learned-sparse).

BGE-M3 emits a dense vector and a sparse lexical-weight map in a single forward
pass. The dense vector drives semantic search; the sparse map is converted to
Qdrant's ``(indices, values)`` sparse-vector form so a single model powers the
hybrid retrieval used by the enhanced pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SparseVector:
    """A sparse vector in index/value form.

    Attributes:
        indices: Non-zero dimension indices (BGE-M3 token ids).
        values: Weights aligned with ``indices``.
    """

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


@dataclass
class TextEmbedding:
    """A text embedding bundling dense and sparse representations.

    Attributes:
        dense: The dense embedding vector.
        sparse: The learned-sparse lexical representation.
    """

    dense: list[float]
    sparse: SparseVector


class TextEmbedder:
    """Wraps the BGE-M3 model for document and query embedding.

    Attributes:
        model_name: Hugging Face model identifier.
        use_fp16: Whether to load weights in half precision.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        """Initialises the embedder.

        Args:
            model_name: Hugging Face model identifier.
            use_fp16: Load weights in FP16 to halve VRAM use (GPU only).
            device: Torch device string; ``None`` lets the library auto-detect.
                FP16 is disabled automatically on CPU.
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device != "cpu"
        self._model: Any = None

    def load(self) -> None:
        """Loads the BGE-M3 weights."""
        from FlagEmbedding import BGEM3FlagModel

        kwargs: dict[str, Any] = {"use_fp16": self.use_fp16}
        if self.device is not None:
            kwargs["devices"] = self.device
        self._model = BGEM3FlagModel(self.model_name, **kwargs)

    def embed(self, texts: list[str], batch_size: int = 16) -> list[TextEmbedding]:
        """Embeds a batch of texts into dense and sparse representations.

        Args:
            texts: Input strings.
            batch_size: Encoder batch size.

        Returns:
            One :class:`TextEmbedding` per input, in input order.
        """
        if self._model is None:
            self.load()
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        embeddings: list[TextEmbedding] = []
        for dense, weights in zip(dense_vecs, lexical_weights):
            sparse = SparseVector(
                indices=[int(token_id) for token_id in weights.keys()],
                values=[float(weight) for weight in weights.values()],
            )
            embeddings.append(
                TextEmbedding(dense=[float(x) for x in dense], sparse=sparse)
            )
        return embeddings

    def embed_query(self, query: str) -> TextEmbedding:
        """Embeds a single query string.

        Args:
            query: The query text.

        Returns:
            The query's :class:`TextEmbedding`.
        """
        return self.embed([query])[0]

    def unload(self) -> None:
        """Releases the model and clears the CUDA cache."""
        self._model = None
        _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    """Releases cached CUDA memory if torch with CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
