#!/usr/bin/env python
"""Diagnostic: run a vision-language model directly on one figure/table crop.

This isolates the VLM from the RAG pipeline. It feeds a *single image* plus a
question straight to the model -- no retrieval, no caption, no other evidence --
so you can see what the model actually reads off the crop, and compare models or
input resolutions side by side. (This is the same image the demo struggled with:
a dense HER2/EC50 table, shipped here as ``dev/her2_table.png``.)

There is no "Qwen3.5-VL"; the pinned ``transformers==4.56.1`` supports the
Qwen2.x-VL line only (it does not recognise ``qwen3-vl``). Compare any ids that
fit your GPU via ``--models``; use ``--load-in-4bit`` (needs the ``quant`` extra)
to fit a 7B on 12 GiB.

Examples:
    python dev/diagnose_vlm.py
    python dev/diagnose_vlm.py --models Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2-VL-7B-Instruct --load-in-4bit
    python dev/diagnose_vlm.py --upscale 0          # feed the native 253x385 crop
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

_DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "her2_table.png")
_FALLBACK_IMAGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "crops", "PMC5384386_2_457455.png"
)

_DEFAULT_QUESTION = (
    "This is a table of HER2 expression (MFI) and EC50 values across cancer cell "
    "lines. Reading directly from the image, give the minimum and maximum HER2 "
    "MFI and the minimum and maximum EC50, naming the cell line at each extreme."
)

_SYSTEM = (
    "You are reading a scientific figure or table. Answer ONLY from the attached "
    "image: read every value, axis label and table cell directly from the pixels. "
    "If a value is too small or blurry to read, say so explicitly rather than "
    "guessing."
)


def _upscale(image, long_side: int):
    """Upscales the image so its long side is at least ``long_side`` (0 = off)."""
    from PIL import Image

    if long_side <= 0 or max(image.size) >= long_side:
        return image
    scale = long_side / max(image.size)
    return image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
    )


def _load(model_id: str, device: str, load_in_4bit: bool):
    """Loads a VLM + processor, optionally in 4-bit."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, AutoProcessor.from_pretrained(model_id)


def _answer(model, processor, image, question: str, max_new_tokens: int) -> str:
    """Runs one image + question through the model and returns the answer text."""
    import torch

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main() -> None:
    """Parses arguments and runs each model on the crop."""
    parser = argparse.ArgumentParser(description="Run VLM(s) directly on one crop.")
    parser.add_argument(
        "--models",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Comma-separated HF model ids to compare (run in order).",
    )
    parser.add_argument("--image", default=None, help="Image path (default: the crop).")
    parser.add_argument("--question", default=_DEFAULT_QUESTION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--upscale",
        type=int,
        default=896,
        help="Upscale the crop's long side to this many px (0 = native).",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    image_path = args.image or (
        _DEFAULT_IMAGE if os.path.exists(_DEFAULT_IMAGE) else _FALLBACK_IMAGE
    )
    image = Image.open(image_path).convert("RGB")
    native = image.size
    image = _upscale(image, args.upscale)

    print(f"image      : {image_path}")
    print(f"resolution : native {native} -> fed {image.size}")
    print(f"question   : {args.question}\n")

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_id in model_ids:
        print("=" * 78)
        print(f"MODEL: {model_id}  (4bit={args.load_in_4bit})")
        print("=" * 78)
        try:
            model, processor = _load(model_id, args.device, args.load_in_4bit)
        except Exception as exc:  # pragma: no cover - diagnostic convenience
            print(f"  [skipped] could not load: {type(exc).__name__}: {exc}\n")
            continue
        answer = _answer(model, processor, image, args.question, args.max_new_tokens)
        print(answer + "\n")

        del model, processor
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":
    main()
