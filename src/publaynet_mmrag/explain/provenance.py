"""Provenance assembly for explainable answers.

Turns an :class:`~publaynet_mmrag.types.Answer` into a structured trace that
records, for every piece of evidence, where it came from, how it was retrieved,
its score, and whether the answer cited it. This is the data the demo renders as
highlighted bounding boxes and a knowledge-graph path list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from publaynet_mmrag.types import Answer


@dataclass
class EvidenceTrace:
    """A single evidence item's provenance.

    Attributes:
        source_id: The chunk or region identifier.
        doc_id: Source document.
        page_index: Source page.
        modality: ``"text"`` or ``"image"``.
        retrieval_source: Channel that surfaced it (dense/hybrid/graph/image).
        score: Final ranking score.
        cited: Whether the generated answer cited this item.
        crop_path: Crop path for image evidence, if any.
        text: The evidence text shown to the model.
    """

    source_id: str
    doc_id: str
    page_index: int
    modality: str
    retrieval_source: str
    score: float
    cited: bool
    crop_path: str = ""
    text: str = ""


@dataclass
class Provenance:
    """The full explainability record for one answer.

    Attributes:
        question: The original query.
        answer: The generated answer text.
        reasoning: The chain-of-thought trace, if surfaced.
        evidence: Per-item provenance traces.
        graph_paths: Knowledge-graph path traces used in retrieval.
    """

    question: str
    answer: str
    reasoning: str
    evidence: list[EvidenceTrace] = field(default_factory=list)
    graph_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialises the provenance record.

        Returns:
            A JSON-compatible dictionary.
        """
        return asdict(self)


def build_provenance(answer: Answer) -> Provenance:
    """Builds a provenance record from a generated answer.

    Args:
        answer: The answer together with its evidence.

    Returns:
        The assembled :class:`Provenance`.
    """
    cited = set(answer.citations)
    traces: list[EvidenceTrace] = []
    for item in answer.evidence:
        source_id = item.chunk_id or item.region_id or ""
        traces.append(
            EvidenceTrace(
                source_id=source_id,
                doc_id=item.doc_id,
                page_index=item.page_index,
                modality=item.modality,
                retrieval_source=item.source,
                score=round(item.score, 4),
                cited=source_id in cited,
                crop_path=item.crop_path or "",
                text=item.text,
            )
        )
    return Provenance(
        question=answer.question,
        answer=answer.text,
        reasoning=answer.reasoning,
        evidence=traces,
        graph_paths=answer.graph_paths,
    )
