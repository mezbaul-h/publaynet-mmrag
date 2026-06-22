"""In-process vision-language model for multimodal answer generation.

A text-only generator can only answer questions about a figure or table from a
*caption* -- a lossy, sometimes hallucinated paraphrase. This wraps a Qwen-VL
checkpoint so the generator can instead read the retrieved crop **pixels**
directly (true visual question answering), while still handling text-only
questions. It loads through the generic ``AutoModelForImageTextToText`` class, so
the same wrapper backs any compatible VLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage


class VisionLLM:
    """Wraps a Qwen-VL checkpoint for grounded multimodal generation.

    Attributes:
        model_name: Hugging Face model identifier.
        device: Torch device string.
        max_new_tokens: Decoding budget.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
    ) -> None:
        """Initialises the vision LLM.

        Args:
            model_name: Hugging Face VLM identifier.
            device: Torch device string.
            max_new_tokens: Decoding budget per answer.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
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

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        images: Optional[list["PILImage"]] = None,
        temperature: float = 0.1,
    ) -> str:
        """Generates an answer from a text prompt and optional images.

        Args:
            system_prompt: The grounding system instruction.
            user_text: The user message (question + formatted text evidence).
            images: Retrieved figure/table crops to attach, or ``None`` for a
                text-only question.
            temperature: Sampling temperature (greedy at 0).

        Returns:
            The generated answer text.
        """
        if self._model is None:
            self.load()
        import torch

        images = images or []
        content: list[dict[str, Any]] = [
            {"type": "image", "image": im} for im in images
        ]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = self._processor(**processor_kwargs).to(self.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[
            0
        ].strip()

    def unload(self) -> None:
        """Releases the model and clears the CUDA cache."""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
