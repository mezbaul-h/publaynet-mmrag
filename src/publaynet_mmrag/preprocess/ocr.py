"""OCR over text-bearing region crops with Surya.

Surya is the sole OCR engine: it is strong on multi-column scientific text and
installs entirely via ``pip`` with weights fetched from the Hub, so no system
binary is required. The engine exposes explicit ``load`` and ``unload`` so
Stage 1 can release GPU memory before the indexing stage loads the embedders --
the models never need to co-reside on the 12 GiB card.

Surya note: the v1.x predictor API used here is
``RecognitionPredictor(FoundationPredictor())`` with a ``DetectionPredictor``,
returning page objects exposing ``text_lines``. Surya v2 (>= 0.20) replaced the
foundation predictor with a VLM inference manager and changed the output schema;
pin ``surya-ocr<0.20`` to use this code unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage


class SuryaOcrEngine:
    """Surya line-level OCR (v1.x predictor API).

    Attributes:
        device: Torch device string used via the ``TORCH_DEVICE`` env var.
        verbose: If ``False`` (default), Surya's own per-page detection /
            recognition progress bars are suppressed so they do not scroll the
            caller's Stage 1 progress bar out of view.
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        detector_batch_size: int = 6,
        recognition_batch_size: int = 32,
    ) -> None:
        """Initialises the engine.

        Args:
            device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
            verbose: Show Surya's per-page progress bars (off by default).
            detector_batch_size: Surya detection batch size. Lower values cut
                peak VRAM, which matters on a 12 GiB card with dense pages.
            recognition_batch_size: Surya recognition batch size.
        """
        self.device = device
        self.verbose = verbose
        self.detector_batch_size = detector_batch_size
        self.recognition_batch_size = recognition_batch_size
        self._recognition: Any = None
        self._detection: Any = None

    def load(self) -> None:
        """Instantiates the detection and recognition predictors."""
        import os

        os.environ.setdefault("TORCH_DEVICE", self.device)
        # Cap Surya's internal batch sizes to keep peak VRAM within a 12 GiB
        # budget on dense pages, and reduce allocator fragmentation. All are
        # Surya/torch settings read from the environment before CUDA init, so
        # they must be set before importing surya / touching the GPU.
        os.environ.setdefault("DETECTOR_BATCH_SIZE", str(self.detector_batch_size))
        os.environ.setdefault(
            "RECOGNITION_BATCH_SIZE", str(self.recognition_batch_size)
        )
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Suppress Surya's internal tqdm bars (Surya-specific setting; read from
        # the environment at import time, so it must be set before importing
        # surya). This does not affect the caller's own tqdm progress bar.
        if not self.verbose:
            os.environ.setdefault("DISABLE_TQDM", "true")
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self._recognition = RecognitionPredictor(FoundationPredictor())
        self._detection = DetectionPredictor()

    def recognise(self, images: list["PILImage"]) -> list[str]:
        """Runs detection + recognition and joins lines in reading order.

        Args:
            images: Region crops to read.

        Returns:
            One newline-joined string per crop.
        """
        if self._recognition is None:
            self.load()
        predictions = self._recognition(images, det_predictor=self._detection)
        outputs: list[str] = []
        for page in predictions:
            lines = [ln.text for ln in page.text_lines if ln.text.strip()]
            outputs.append("\n".join(lines).strip())
        # Release per-page activations so fragmentation does not accumulate
        # across thousands of pages.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
        return outputs

    def unload(self) -> None:
        """Drops predictor references and clears the CUDA cache."""
        self._recognition = None
        self._detection = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def build_ocr_engine(
    device: str = "cuda",
    verbose: bool = False,
    detector_batch_size: int = 6,
    recognition_batch_size: int = 32,
) -> SuryaOcrEngine:
    """Constructs the Surya OCR engine.

    Args:
        device: Torch device string for the predictors.
        verbose: Show Surya's per-page progress bars (off by default).
        detector_batch_size: Surya detection batch size (lower = less VRAM).
        recognition_batch_size: Surya recognition batch size.

    Returns:
        A configured :class:`SuryaOcrEngine`.
    """
    return SuryaOcrEngine(
        device=device,
        verbose=verbose,
        detector_batch_size=detector_batch_size,
        recognition_batch_size=recognition_batch_size,
    )
