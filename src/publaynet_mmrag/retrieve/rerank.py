"""Cross-encoder reranking with bge-reranker-v2-m3.

Reranking is an ablation knob: it sharpens the candidate ordering by scoring
each (query, passage) pair jointly. It is enabled only in the enhanced
configuration so its contribution can be measured independently.
"""

from __future__ import annotations

from typing import Any


class Reranker:
    """Wraps the BGE reranker cross-encoder.

    Attributes:
        model_name: Hugging Face reranker identifier.
        use_fp16: Whether to load weights in half precision.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        """Initialises the reranker.

        Args:
            model_name: Hugging Face reranker identifier.
            use_fp16: Load weights in FP16 to reduce VRAM use (GPU only).
            device: Torch device string; ``None`` lets the library auto-detect.
                FP16 is disabled automatically on CPU.
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device != "cpu"
        self._model: Any = None

    def load(self) -> None:
        """Loads the reranker weights."""
        from FlagEmbedding import FlagReranker

        kwargs: dict[str, Any] = {"use_fp16": self.use_fp16}
        if self.device is not None:
            kwargs["devices"] = self.device
        self._model = FlagReranker(self.model_name, **kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Scores each passage against the query.

        Args:
            query: The query text.
            passages: Candidate passage texts.

        Returns:
            One relevance score per passage, in input order.
        """
        if self._model is None:
            self.load()
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        from publaynet_mmrag.quiet import silence_stderr

        with silence_stderr():
            scores = self._model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            return [scores]
        return [float(s) for s in scores]

    def unload(self) -> None:
        """Releases the model and clears the CUDA cache."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
