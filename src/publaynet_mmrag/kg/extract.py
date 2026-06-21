"""Entity and relation extraction for the knowledge graph.

Entities are extracted with GLiNER, a zero-shot NER model whose label set is
configurable (methods, datasets, metrics, results, authors, tasks, models).
Relations are extracted with the local LLM as ``(subject, relation, object)``
triples; this is optional and falls back to co-occurrence edges only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Entity:
    """A typed entity mention.

    Attributes:
        text: The surface form of the entity.
        label: The entity type (one of the configured labels).
        score: Model confidence in ``[0, 1]``.
    """

    text: str
    label: str
    score: float


@dataclass
class Triple:
    """A directed ``(subject, relation, object)`` relation.

    Attributes:
        subject: The head entity surface form.
        relation: The relation type.
        object: The tail entity surface form.
    """

    subject: str
    relation: str
    object: str


class EntityExtractor:
    """Extracts typed entities with GLiNER.

    Attributes:
        model_name: Hugging Face GLiNER model identifier.
        labels: The entity types to extract.
        threshold: Minimum confidence to keep a mention.
    """

    def __init__(
        self,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
        window_words: int = 250,
        overlap_words: int = 30,
    ) -> None:
        """Initialises the extractor.

        Args:
            model_name: Hugging Face GLiNER model identifier.
            labels: Entity types to extract.
            threshold: Minimum confidence to keep a mention.
            window_words: Word budget per NER window. GLiNER truncates inputs to
                384 tokens, so long passages are split into windows below that
                limit to avoid silently dropping entities in the tail.
            overlap_words: Word overlap between consecutive windows, so an
                entity straddling a boundary is still seen whole in one window.
        """
        self.model_name = model_name
        self.labels = labels
        self.threshold = threshold
        self.window_words = window_words
        self.overlap_words = overlap_words
        self._model: Any = None

    def load(self) -> None:
        """Loads the GLiNER model."""
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(self.model_name)

    def _windows(self, text: str) -> list[str]:
        """Splits text into overlapping word windows below GLiNER's limit.

        Args:
            text: The passage to split.

        Returns:
            One window for short passages, or several overlapping windows for
            passages longer than ``window_words``.
        """
        words = text.split()
        if len(words) <= self.window_words:
            return [text]
        step = max(1, self.window_words - self.overlap_words)
        windows: list[str] = []
        for start in range(0, len(words), step):
            windows.append(" ".join(words[start : start + self.window_words]))
            if start + self.window_words >= len(words):
                break
        return windows

    def extract(self, text: str) -> list[Entity]:
        """Extracts entities from a passage.

        Long passages are processed in overlapping windows so that entities
        beyond GLiNER's 384-token input limit are not lost; results are unioned
        and de-duplicated.

        Args:
            text: The passage to analyse.

        Returns:
            Deduplicated entity mentions above the confidence threshold.
        """
        if self._model is None:
            self.load()
        from publaynet_mmrag.quiet import silence_stderr

        seen: set[tuple[str, str]] = set()
        entities: list[Entity] = []
        with silence_stderr():
            for window in self._windows(text):
                spans = self._model.predict_entities(
                    window, self.labels, threshold=self.threshold
                )
                for span in spans:
                    key = (span["text"].lower().strip(), span["label"])
                    if key in seen or not key[0]:
                        continue
                    seen.add(key)
                    entities.append(
                        Entity(
                            text=span["text"].strip(),
                            label=span["label"],
                            score=span["score"],
                        )
                    )
        return entities

    def unload(self) -> None:
        """Releases the model and clears the CUDA cache."""
        self._model = None
        _empty_cuda_cache()


_RELATION_PROMPT = (
    "Extract factual relationships from the scientific text as JSON. "
    "Return only a JSON array of objects with keys 'subject', 'relation', "
    "'object'. Use concise relation verbs (e.g. 'uses', 'evaluated_on', "
    "'achieves', 'proposes'). Text:\n\n{text}\n\nJSON:"
)


class RelationExtractor:
    """Extracts relation triples using the shared in-process LLM.

    Attributes:
        llm: The shared language model.
    """

    def __init__(self, llm: "Any") -> None:
        """Initialises the extractor.

        Args:
            llm: A :class:`~publaynet_mmrag.reason.llm.LocalLLM` instance.
        """
        self.llm = llm

    def extract(self, text: str) -> list[Triple]:
        """Extracts relation triples from a passage.

        Args:
            text: The passage to analyse.

        Returns:
            The parsed triples; an empty list if parsing fails.
        """
        content = self.llm.chat(
            messages=[{"role": "user", "content": _RELATION_PROMPT.format(text=text)}],
            temperature=0.0,
        )
        return _parse_triples(content)


def _parse_triples(content: str) -> list[Triple]:
    """Parses a JSON triple array from possibly noisy LLM output.

    Args:
        content: Raw model output that should contain a JSON array.

    Returns:
        Parsed triples, ignoring malformed entries.
    """
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        rows = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return []
    triples: list[Triple] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        subject = str(row.get("subject", "")).strip()
        relation = str(row.get("relation", "")).strip()
        obj = str(row.get("object", "")).strip()
        if subject and relation and obj:
            triples.append(Triple(subject=subject, relation=relation, object=obj))
    return triples


def _empty_cuda_cache() -> None:
    """Releases cached CUDA memory if torch with CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
