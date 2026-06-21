"""Synthetic evaluation-set construction.

PubLayNet ships no QA pairs, so a grounded set is synthesised: the shared LLM is
asked to write a question answerable from a single sampled chunk, and that
chunk's id becomes the gold context. This yields known-gold retrieval targets
(for Recall@k, MRR, nDCG) without manual annotation. The same LLM and chunks are
used for both pipelines, so the comparison is fair.
"""

from __future__ import annotations

import random
from typing import Any

from publaynet_mmrag.reason.llm import LocalLLM
from publaynet_mmrag.types import Chunk

_QUESTION_PROMPT = (
    "Read the scientific passage below and write one specific question that is "
    "answerable using only this passage. Return just the question, no preamble.\n\n"
    "Passage:\n{text}\n\nQuestion:"
)


def synthesise_qa(
    chunks: list[Chunk],
    num_questions: int,
    llm: LocalLLM,
    seed: int = 42,
    min_chars: int = 200,
) -> list[dict[str, Any]]:
    """Generates grounded question/gold-chunk pairs.

    Args:
        chunks: Candidate source chunks.
        num_questions: Number of questions to generate.
        llm: The shared in-process language model.
        seed: Sampling seed for reproducibility.
        min_chars: Minimum chunk length to be eligible as a source.

    Returns:
        Rows with keys ``question``, ``gold_chunk_id``, ``gold_doc_id`` and
        ``gold_page_index``.
    """
    rng = random.Random(seed)

    eligible = [c for c in chunks if len(c.text) >= min_chars]
    if not eligible:
        return []
    sample = rng.sample(eligible, min(num_questions, len(eligible)))

    rows: list[dict[str, Any]] = []
    from tqdm import tqdm

    for chunk in tqdm(sample, desc="QA synthesis", unit="q"):
        question = llm.chat(
            messages=[
                {"role": "user", "content": _QUESTION_PROMPT.format(text=chunk.text)}
            ],
            temperature=0.2,
        ).strip().strip('"')
        if not question:
            continue
        rows.append(
            {
                "question": question,
                "gold_chunk_id": chunk.chunk_id,
                "gold_doc_id": chunk.doc_id,
                "gold_page_index": chunk.page_index,
            }
        )
    return rows
