"""Generation-quality metrics via a lightweight local LLM judge.

This replaces the external RAGAS + LangChain dependency stack with a small,
self-contained judge that reuses the pipeline's own in-process LLM. Two scores
are produced per answer:

* **faithfulness** -- is the answer supported by the retrieved contexts?
* **answer relevancy** -- does the answer address the question?

Each is elicited as a single number in ``[0, 1]``; parsing is defensive and a
sample that cannot be scored is skipped rather than failing the run. This is a
deliberately simple judge suited to a proof-of-concept; swap in a stronger
evaluator if needed.
"""

from __future__ import annotations

import re
from typing import Any

from publaynet_mmrag.reason.llm import LocalLLM

_FAITHFULNESS_PROMPT = (
    "You are grading whether an answer is supported by the given context. "
    "Reply with a single number from 0 to 1, where 1 means every claim in the "
    "answer is supported by the context and 0 means none is.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}\n\nScore:"
)

_RELEVANCY_PROMPT = (
    "You are grading whether an answer addresses the question. Reply with a "
    "single number from 0 to 1, where 1 means the answer fully addresses the "
    "question and 0 means it is unrelated.\n\n"
    "Question:\n{question}\n\nAnswer:\n{answer}\n\nScore:"
)

_NUMBER_RE = re.compile(r"(\d*\.?\d+)")


def _score(llm: LocalLLM, prompt: str) -> float | None:
    """Elicits and parses a single 0-1 score from the judge.

    Args:
        llm: The shared language model.
        prompt: The fully formatted grading prompt.

    Returns:
        The parsed score clamped to ``[0, 1]``, or ``None`` if unparseable.
    """
    reply = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.0)
    match = _NUMBER_RE.search(reply)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, value))


def evaluate_generation(
    samples: list[dict[str, Any]], llm: LocalLLM, desc: str = "judge"
) -> dict[str, float]:
    """Scores generated answers for faithfulness and relevancy.

    Args:
        samples: Rows with keys ``question``, ``answer`` and ``contexts``
            (a list of evidence strings).
        llm: The shared in-process language model used as the judge.
        desc: Label for the progress bar.

    Returns:
        Mean ``faithfulness`` and ``answer_relevancy`` over the scorable
        samples; an empty dict if nothing could be scored.
    """
    if not samples:
        return {}

    from tqdm import tqdm

    faith: list[float] = []
    relevancy: list[float] = []
    for sample in tqdm(samples, desc=desc, unit="ans"):
        context = "\n\n".join(sample.get("contexts", []))
        f = _score(
            llm,
            _FAITHFULNESS_PROMPT.format(context=context, answer=sample["answer"]),
        )
        r = _score(
            llm,
            _RELEVANCY_PROMPT.format(
                question=sample["question"], answer=sample["answer"]
            ),
        )
        if f is not None:
            faith.append(f)
        if r is not None:
            relevancy.append(r)

    metrics: dict[str, float] = {}
    if faith:
        metrics["faithfulness"] = sum(faith) / len(faith)
    if relevancy:
        metrics["answer_relevancy"] = sum(relevancy) / len(relevancy)
    return metrics
