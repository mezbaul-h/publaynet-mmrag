"""Image and text embedding with SigLIP2.

SigLIP2 provides aligned image and text towers, so figure and table crops can
be embedded for visual retrieval and a natural-language query can be embedded
into the same space for text-to-image search. Embeddings are L2-normalised so
cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage


class ImageEmbedder:
    """Wraps a SigLIP2 checkpoint for image and text embedding.

    Attributes:
        model_name: Hugging Face model identifier.
        device: Torch device string.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str = "cuda",
    ) -> None:
        """Initialises the embedder.

        Args:
            model_name: Hugging Face SigLIP2 model identifier.
            device: Torch device string.
        """
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._max_text_len = 64  # SigLIP text tower limit; refined at load().

    def load(self) -> None:
        """Loads the SigLIP2 model and processor onto the device."""
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModel.from_pretrained(self.model_name, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        text_config = getattr(self._model.config, "text_config", None)
        self._max_text_len = getattr(text_config, "max_position_embeddings", 64)

    def embed_images(
        self, images: list["PILImage"], batch_size: int = 16
    ) -> list[list[float]]:
        """Embeds image crops into the shared space.

        Args:
            images: Region crops to embed.
            batch_size: Forward-pass batch size.

        Returns:
            One L2-normalised embedding per image, in input order.
        """
        if self._model is None:
            self.load()
        import torch

        vectors: list[list[float]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
            vectors.extend(features.float().cpu().tolist())
        return vectors

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embeds text into the shared image/text space.

        Args:
            texts: Query or caption strings.

        Returns:
            One L2-normalised embedding per text, in input order.
        """
        if self._model is None:
            self.load()
        import torch

        inputs = self._processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self._max_text_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().tolist()

    def unload(self) -> None:
        """Releases the model and clears the CUDA cache."""
        self._model = None
        self._processor = None
        _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    """Releases cached CUDA memory if torch with CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
