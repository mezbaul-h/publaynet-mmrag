"""Prompt templates for grounded, citable answer generation.

The system prompt forces the model to answer only from the supplied evidence and
to cite the bracketed source tags, which both grounds the answer and yields the
citation list the explainability stage maps back to bounding boxes.
"""

from __future__ import annotations

from publaynet_mmrag.types import RetrievedItem

SYSTEM_PROMPT = (
    "You are a scientific question-answering assistant. Answer the question "
    "using only the numbered evidence provided. Cite every claim with the "
    "matching source tag in square brackets, e.g. [S1]. If the evidence is "
    "insufficient, say so explicitly. Do not use outside knowledge."
)

COT_INSTRUCTION = (
    "First reason step by step under a 'Reasoning:' heading, then give the "
    "final answer under an 'Answer:' heading."
)

DIRECT_INSTRUCTION = "Give the answer under an 'Answer:' heading."


def format_evidence(items: list[RetrievedItem]) -> tuple[str, dict[str, str]]:
    """Renders retrieved items as numbered, tagged evidence blocks.

    Args:
        items: The retrieved items to include as context.

    Returns:
        A tuple of the formatted evidence string and a mapping from source tag
        (e.g. ``"S1"``) to the underlying chunk or region identifier.
    """
    lines: list[str] = []
    tag_map: dict[str, str] = {}
    for i, item in enumerate(items, start=1):
        tag = f"S{i}"
        source_id = item.chunk_id or item.region_id or f"item-{i}"
        tag_map[tag] = source_id
        locator = f"doc {item.doc_id}, page {item.page_index}, {item.modality}"
        lines.append(f"[{tag}] ({locator})\n{item.text.strip()}")
    return "\n\n".join(lines), tag_map


def build_user_prompt(
    question: str, evidence: str, chain_of_thought: bool
) -> str:
    """Assembles the user prompt from the question and evidence.

    Args:
        question: The natural-language question.
        evidence: The formatted evidence string.
        chain_of_thought: Whether to request step-by-step reasoning.

    Returns:
        The complete user prompt.
    """
    instruction = COT_INSTRUCTION if chain_of_thought else DIRECT_INSTRUCTION
    return (
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"{instruction}"
    )
