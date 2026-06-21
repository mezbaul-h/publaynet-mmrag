"""Optional figure/table captioning with a vision-language model.

Captions give visual regions a text handle: they can be embedded by the text
model, indexed for the figure-retrieval extension, and shown in the
explainability view. This is the heaviest model in the pipeline, so it runs only
in Stage 1, never co-resident with the serving models, and can be skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage

_DEFAULT_PROMPT = (
    "Describe this scientific figure or table in one concise sentence, naming "
    "what it depicts and any axes, variables or quantities shown."
)


class Captioner:
    """Captions image crops using a Qwen-VL checkpoint.

    Loaded through the generic ``AutoModelForImageTextToText`` class so any
    Qwen-VL generation (or other compatible VLM) works without code changes.

    Attributes:
        model_name: Hugging Face model identifier.
        device: Torch device string.
        prompt: Instruction prepended to each image.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        prompt: str = _DEFAULT_PROMPT,
    ) -> None:
        """Initialises the captioner.

        Args:
            model_name: Hugging Face VLM identifier.
            device: Torch device string.
            prompt: Instruction text for captioning.
        """
        self.model_name = model_name
        self.device = device
        self.prompt = prompt
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Loads the VLM and its processor onto the device."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModelForImageTextToText.from_pretrained(self.model_name, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def caption(self, image: "PILImage", max_new_tokens: int = 64) -> str:
        """Generates a one-line caption for a single image.

        Args:
            image: The region crop to caption.
            max_new_tokens: Decoding budget for the caption.

        Returns:
            The generated caption text.
        """
        if self._model is None:
            self.load()
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[
            0
        ].strip()

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
