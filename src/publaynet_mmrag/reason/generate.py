"""Answer generation over retrieved evidence.

Generation uses the shared in-process :class:`~publaynet_mmrag.reason.llm.LocalLLM`
(Transformers), so no external inference server is required. The system prompt
forces grounded, citable answers; citations are parsed back to source ids for
the explainability stage.
"""

from __future__ import annotations

import re

from publaynet_mmrag.config import GenerationConfig
from publaynet_mmrag.reason import prompts
from publaynet_mmrag.reason.llm import LocalLLM
from publaynet_mmrag.types import Answer, RetrievedItem

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
