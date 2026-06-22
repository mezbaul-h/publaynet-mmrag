"""Answer generation over retrieved evidence.

Generation uses the shared in-process :class:`~publaynet_mmrag.reason.llm.LocalLLM`
(Transformers), so no external inference server is required. The system prompt
forces grounded, citable answers; citations are parsed back to source ids for
the explainability stage.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from publaynet_mmrag.config import GenerationConfig
from publaynet_mmrag.reason import prompts
from publaynet_mmrag.reason.llm import LocalLLM
from publaynet_mmrag.reason.vision_llm import VisionLLM
from publaynet_mmrag.types import Answer, RetrievedItem

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage

_CITATION_RE = re.compile(r"\[(S\d+)\]")


class Generator:
    """Generates grounded answers from retrieved evidence.

    Attributes:
        llm: The shared language model.
        generation: Generation configuration.
    """

    def __init__(self, llm: LocalLLM, generation: GenerationConfig) -> None:
        """Initialises the generator.

        Args:
            llm: The shared in-process language model.
            generation: Generation configuration.
        """
        self.llm = llm
        self.generation = generation

    def generate(
        self, question: str, items: list[RetrievedItem], graph_paths: list[str]
    ) -> Answer:
        """Generates an answer grounded in the retrieved evidence.

        Args:
            question: The natural-language question.
            items: Retrieved evidence items, already ranked.
            graph_paths: Knowledge-graph path traces used during retrieval.

        Returns:
            The :class:`~publaynet_mmrag.types.Answer`, with reasoning split out
            and citations resolved back to source identifiers.
        """
        items = items[: self.generation.max_context_items]
        evidence, tag_map = prompts.format_evidence(items)
        user_prompt = prompts.build_user_prompt(
            question, evidence, self.generation.chain_of_thought
        )

        content = self.llm.chat(
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.generation.temperature,
        )
        reasoning, answer_text = _split_reasoning(content)

        cited_tags = sorted(set(_CITATION_RE.findall(content)))
        citations = [tag_map[tag] for tag in cited_tags if tag in tag_map]

        return Answer(
            question=question,
            text=answer_text,
            reasoning=reasoning,
            citations=citations,
            evidence=items,
            graph_paths=graph_paths,
        )


_VISION_NOTE = (
    "The attached images are the figure/table evidence items, in the order they "
    "appear below. They have no text description on purpose: read every value, "
    "axis label and table cell you report directly from the images. Do not guess "
    "numbers you cannot read; say so instead."
)

_MIN_VLM_LONG_SIDE = 896  # Upscale small crops so the VLM has pixels to read.


def _prep_crop(image: "PILImage") -> "PILImage":
    """Upscales a small crop so dense figures/tables are more legible to the VLM.

    PubLayNet region crops are taken at the page's native (often low) resolution;
    a dense table at ~250 px is hard for any reader. Upscaling adds no real
    detail but gives the vision encoder a larger grid to work with, which helps
    it read small glyphs.

    Args:
        image: The region crop.

    Returns:
        The crop, upscaled so its long side is at least ``_MIN_VLM_LONG_SIDE``.
    """
    from PIL import Image

    long_side = max(image.size)
    if long_side >= _MIN_VLM_LONG_SIDE:
        return image
    scale = _MIN_VLM_LONG_SIDE / long_side
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


def _resolve_crop(crop_path: str) -> str | None:
    """Resolves a stored (possibly repo-relative) crop path to an existing file.

    Args:
        crop_path: The ``crop_path`` recorded on a retrieved image item.

    Returns:
        A path to the crop if it exists (as-is or relative to the CWD), else
        ``None``.
    """
    if not crop_path:
        return None
    for candidate in (crop_path, os.path.abspath(crop_path)):
        if os.path.exists(candidate):
            return candidate
    return None


class VisionGenerator:
    """Generates answers with a VLM that sees the retrieved figure/table crops.

    Mirrors :class:`Generator`'s interface but routes generation through a
    vision-language model, attaching the retrieved crops so the model reads
    figures and tables from pixels rather than from a lossy caption. Text-only
    questions (no image evidence) are answered by the same VLM with no images.

    Attributes:
        vision_llm: The in-process vision-language model.
        generation: Generation configuration.
        max_images: Maximum crops to attach per answer.
    """

    def __init__(
        self,
        vision_llm: VisionLLM,
        generation: GenerationConfig,
        max_images: int = 4,
    ) -> None:
        """Initialises the vision generator.

        Args:
            vision_llm: The in-process vision-language model.
            generation: Generation configuration.
            max_images: Maximum crops to attach per answer (VLM input bound).
        """
        self.vision_llm = vision_llm
        self.generation = generation
        self.max_images = max_images

    def generate(
        self, question: str, items: list[RetrievedItem], graph_paths: list[str]
    ) -> Answer:
        """Generates a grounded answer, attaching retrieved crops to the VLM.

        Args:
            question: The natural-language question.
            items: Retrieved evidence items, already ranked.
            graph_paths: Knowledge-graph path traces used during retrieval.

        Returns:
            The :class:`~publaynet_mmrag.types.Answer`.
        """
        import dataclasses

        from PIL import Image

        items = items[: self.generation.max_context_items]

        # Attach crops for image evidence, and blank out their (possibly wrong)
        # caption text in the prompt so the VLM reads the image, not the caption.
        images = []
        display: list[RetrievedItem] = []
        for item in items:
            resolved = (
                _resolve_crop(item.crop_path or "")
                if item.modality == "image" and len(images) < self.max_images
                else None
            )
            if resolved:
                images.append(_prep_crop(Image.open(resolved).convert("RGB")))
                display.append(
                    dataclasses.replace(
                        item, text="[figure/table — see attached image]"
                    )
                )
            else:
                display.append(item)

        evidence, tag_map = prompts.format_evidence(display)
        user_prompt = prompts.build_user_prompt(
            question, evidence, self.generation.chain_of_thought
        )
        if images:
            user_prompt = f"{_VISION_NOTE}\n\n{user_prompt}"

        content = self.vision_llm.generate(
            system_prompt=prompts.SYSTEM_PROMPT,
            user_text=user_prompt,
            images=images,
            temperature=self.generation.temperature,
        )
        reasoning, answer_text = _split_reasoning(content)
        cited_tags = sorted(set(_CITATION_RE.findall(content)))
        citations = [tag_map[tag] for tag in cited_tags if tag in tag_map]

        return Answer(
            question=question,
            text=answer_text,
            reasoning=reasoning,
            citations=citations,
            evidence=items,
            graph_paths=graph_paths,
        )


def _split_reasoning(content: str) -> tuple[str, str]:
    """Splits model output into reasoning and answer sections.

    Args:
        content: The raw model output.

    Returns:
        A ``(reasoning, answer)`` tuple. If no 'Answer:' heading is present the
        whole output is treated as the answer.
    """
    lower = content.lower()
    idx = lower.rfind("answer:")
    if idx == -1:
        return "", content.strip()
    answer = content[idx + len("answer:") :].strip()
    reasoning = content[:idx]
    reasoning = re.sub(r"(?i)reasoning:", "", reasoning).strip()
    return reasoning, answer
