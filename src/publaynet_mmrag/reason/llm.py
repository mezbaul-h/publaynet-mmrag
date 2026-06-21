"""In-process language model over Hugging Face Transformers.

A single :class:`LocalLLM` instance backs every LLM use in the pipeline --
answer generation, relation extraction, QA synthesis and the evaluation judge --
so the weights are loaded once and shared rather than reloaded per component.
This removes any external inference server (the proof-of-concept needs only
``pip install`` and an in-process model whose weights are fetched from the Hub).

The default model is a 3B instruct checkpoint in FP16, which fits a 12 GiB GPU
alongside the CPU-resident retrieval models. A larger 4-bit model can be enabled
via ``load_in_4bit`` (needs the optional ``bitsandbytes`` dependency).
"""

from __future__ import annotations

from typing import Any


def _ensure_bitsandbytes() -> None:
    """Checks that the optional ``bitsandbytes`` dependency is installed.

    Raises:
        ImportError: With install guidance if ``bitsandbytes`` is missing. This
            fails fast with an actionable message rather than letting a raw
            import error surface from deep inside Transformers.
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "llm_load_in_4bit is set but the 'bitsandbytes' package is not "
            "installed. Install the optional quant extra:\n"
            '    pip install -e ".[quant]"\n'
            "or set models.llm_load_in_4bit: false to load in FP16."
        ) from exc


class LocalLLM:
    """A chat-capable causal language model loaded in-process.

    Attributes:
        model_name: Hugging Face model id.
        device: Torch device string (ignored when ``load_in_4bit`` uses
            ``device_map``).
        dtype: Torch dtype name for non-quantised loading.
        load_in_4bit: Whether to load 4-bit weights via bitsandbytes.
        max_new_tokens: Default decoding budget.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        """Initialises the wrapper without loading weights.

        Args:
            model_name: Hugging Face model id.
            device: Torch device string.
            dtype: Torch dtype name for non-quantised loading.
            load_in_4bit: Load 4-bit weights via bitsandbytes.
            max_new_tokens: Default decoding budget.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Loads the tokenizer and model onto the target device."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.load_in_4bit:
            _ensure_bitsandbytes()
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=quant, device_map=self.device
            ).eval()
        else:
            torch_dtype = "auto" if self.dtype == "auto" else getattr(torch, self.dtype)
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch_dtype)
                .to(self.device)
                .eval()
            )

        # The pipeline decodes greedily (deterministic, reproducible output) for
        # every LLM call, so clear the checkpoint's sampling defaults. This makes
        # the generation config agree with the actual behaviour and stops the
        # "generation flags not valid under do_sample=False" warning. It changes
        # no output; greedy decoding already overrode these values.
        gen_config = getattr(self._model, "generation_config", None)
        if gen_config is not None:
            gen_config.do_sample = False
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(gen_config, attr):
                    setattr(gen_config, attr, None)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generates a chat completion for a message list.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.
            temperature: Sampling temperature; ``0`` selects greedy decoding.
            max_new_tokens: Override for the decoding budget.

        Returns:
            The decoded assistant response text.
        """
        if self._model is None:
            self.load()
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        do_sample = temperature and temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": bool(do_sample),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            generated = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        """Releases the model and clears the CUDA cache."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
