"""Synthetic evaluation-set construction (segmented by question type).

PubLayNet ships no QA pairs, so a grounded set is synthesised. To isolate what
each enhanced channel contributes, three *typed* splits are produced, each with
the gold target the relevant channel is built to retrieve:

* ``text``     -- a question answerable from one text chunk; gold = that chunk.
                  Measures text retrieval (dense / hybrid / rerank).
* ``visual``   -- a question answerable from one figure/table; gold = that
                  region. Only the image channel can retrieve it.
* ``multihop`` -- a question that names entity *A* but is answered by a chunk
                  about a graph-neighbour *B* that does not mention *A*; gold =
                  that bridge chunk. Constructed so single-vector text retrieval
                  misses it but graph expansion reaches it -- i.e. questions that
                  genuinely *require* multi-hop retrieval. Membership is verified
                  empirically (dense-miss AND graph-hit), so a noisy graph cannot
                  admit an invalid pair.

The same LLM and corpus back every split, so the baseline-vs-enhanced comparison
stays fair.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterator

import networkx as nx

from publaynet_mmrag.kg.build import CHUNK, ENTITY
from publaynet_mmrag.reason.llm import LocalLLM
from publaynet_mmrag.types import Chunk, Region

_QUESTION_PROMPT = (
    "Read the scientific passage below and write one specific question that is "
    "answerable using only this passage. Return just the question, no preamble.\n\n"
    "Passage:\n{text}\n\nQuestion:"
)

_VISUAL_PROMPT = (
    "Below is a one-line description of a scientific figure or table. Write one "
    "specific question that asks about what this figure/table shows and is "
    "answerable from it. Return just the question, no preamble.\n\n"
    "Figure/table description:\n{caption}\n\nQuestion:"
)

_MULTIHOP_PROMPT = (
    "Read the scientific passage below. Write one specific question that is "
    "answerable using only this passage, but refer to its subject indirectly as "
    '"the {b_label} associated with {a}" instead of naming it directly. The '
    "question must require knowing that connection. Return just the question.\n\n"
    "Passage:\n{text}\n\nQuestion:"
)

_MIN_ENTITY_CHARS = 4


def _ask(llm: LocalLLM, prompt: str) -> str:
    """Asks the LLM for a single question and cleans the reply.

    Args:
        llm: The shared language model.
        prompt: The fully formatted prompt.

    Returns:
        The stripped question text (empty if the model returned nothing).
    """
    return (
        llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.2)
        .strip()
        .strip('"')
    )


def _row(
    question: str,
    qtype: str,
    gold_id: str,
    gold_kind: str,
    gold_doc_id: str,
    gold_page_index: int,
) -> dict[str, Any]:
    """Builds a normalised QA row.

    Args:
        question: The synthesised question.
        qtype: One of ``text``, ``visual`` or ``multihop``.
        gold_id: The gold chunk or region id.
        gold_kind: ``chunk`` or ``region``.
        gold_doc_id: The gold document id.
        gold_page_index: The gold page index.

    Returns:
        A row carrying the normalised gold plus a back-compatible
        ``gold_chunk_id`` (empty for region gold).
    """
    return {
        "question": question,
        "qtype": qtype,
        "gold_id": gold_id,
        "gold_kind": gold_kind,
        "gold_doc_id": gold_doc_id,
        "gold_page_index": gold_page_index,
        "gold_chunk_id": gold_id if gold_kind == "chunk" else "",
    }


def synthesise_text_qa(
    chunks: list[Chunk],
    num_questions: int,
    llm: LocalLLM,
    seed: int = 42,
    min_chars: int = 200,
) -> list[dict[str, Any]]:
    """Generates ``text`` questions, each answerable from one chunk.

    Args:
        chunks: Candidate source chunks.
        num_questions: Number of questions to generate.
        llm: The shared in-process language model.
        seed: Sampling seed for reproducibility.
        min_chars: Minimum chunk length to be eligible as a source.

    Returns:
        Normalised ``text`` rows with chunk gold.
    """
    from tqdm import tqdm

    rng = random.Random(seed)
    eligible = [c for c in chunks if len(c.text) >= min_chars]
    if not eligible:
        return []
    sample = rng.sample(eligible, min(num_questions, len(eligible)))

    rows: list[dict[str, Any]] = []
    for chunk in tqdm(sample, desc="QA synthesis [text]", unit="q"):
        question = _ask(llm, _QUESTION_PROMPT.format(text=chunk.text))
        if question:
            rows.append(
                _row(
                    question,
                    "text",
                    chunk.chunk_id,
                    "chunk",
                    chunk.doc_id,
                    chunk.page_index,
                )
            )
    return rows


def synthesise_visual_qa(
    regions: list[Region],
    num_questions: int,
    llm: LocalLLM,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generates ``visual`` questions, each answerable from one figure/table.

    Args:
        regions: Candidate regions; only captioned visual regions are used.
        num_questions: Number of questions to generate.
        llm: The shared in-process language model.
        seed: Sampling seed for reproducibility.

    Returns:
        Normalised ``visual`` rows with region gold.
    """
    from tqdm import tqdm

    rng = random.Random(seed)
    eligible = [r for r in regions if r.caption and r.caption.strip()]
    if not eligible:
        return []
    sample = rng.sample(eligible, min(num_questions, len(eligible)))

    rows: list[dict[str, Any]] = []
    for region in tqdm(sample, desc="QA synthesis [visual]", unit="q"):
        question = _ask(llm, _VISUAL_PROMPT.format(caption=region.caption))
        if question:
            rows.append(
                _row(
                    question,
                    "visual",
                    region.region_id,
                    "region",
                    region.doc_id,
                    region.page_index,
                )
            )
    return rows


def _multihop_candidates(
    graph: nx.MultiDiGraph,
    chunk_lookup: dict[str, Chunk],
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    """Yields ``(A, B, bridge chunk)`` candidates for multi-hop questions.

    For an entity *B* mentioned in several chunks, a neighbour entity *A* is
    found that co-occurs with *B* in one chunk but is absent from another chunk
    *C2* (also about *B*). *C2* becomes the bridge: a question anchored on *A* but
    answered by *C2* requires hopping A -> B -> C2, which single-vector retrieval
    on the *A* query tends to miss.

    Args:
        graph: The knowledge graph.
        chunk_lookup: Map of chunk id -> :class:`Chunk` (text/doc/page source).
        rng: Seeded RNG controlling iteration order.

    Yields:
        Dicts with ``a``, ``a_label``, ``b``, ``b_label`` and ``chunk`` (the
        bridge :class:`Chunk`).
    """
    entities = [
        n
        for n, d in graph.nodes(data=True)
        if d.get("ntype") == ENTITY
        and len(str(d.get("name", "")).strip()) >= _MIN_ENTITY_CHARS
    ]
    rng.shuffle(entities)

    for b in entities:
        chunks_b = [
            p
            for p in graph.predecessors(b)
            if graph.nodes[p].get("ntype") == CHUNK and p in chunk_lookup
        ]
        if len(chunks_b) < 2:
            continue
        docs = {c: graph.nodes[c].get("doc_id") for c in chunks_b}
        b_name = str(graph.nodes[b].get("name", "")).strip()

        for c1 in chunks_b:
            # Entities co-mentioned with B in C1 are A candidates.
            a_nodes = [
                e
                for e in graph.successors(c1)
                if graph.nodes[e].get("ntype") == ENTITY
                and e != b
                and len(str(graph.nodes[e].get("name", "")).strip())
                >= _MIN_ENTITY_CHARS
            ]
            rng.shuffle(a_nodes)
            for a in a_nodes:
                a_name = str(graph.nodes[a].get("name", "")).strip()
                if a_name.lower() == b_name.lower():
                    continue
                # A bridge chunk about B that does NOT mention A, ideally a
                # different document so the hop crosses documents.
                for c2 in chunks_b:
                    if c2 == c1 or graph.has_edge(c2, a):
                        continue
                    if docs.get(c2) == docs.get(c1):
                        continue
                    yield {
                        "a": a_name,
                        "a_label": graph.nodes[a].get("label", "concept"),
                        "b": b_name,
                        "b_label": graph.nodes[b].get("label", "concept"),
                        "chunk": chunk_lookup[c2],
                    }
                    break
                else:
                    continue
                break  # one candidate per (B, C1) keeps the pool diverse


def synthesise_multihop_qa(
    graph: nx.MultiDiGraph,
    chunks: list[Chunk],
    num_questions: int,
    llm: LocalLLM,
    dense_miss_fn: Callable[[str, str], bool],
    graph_reach_fn: Callable[[str, str], bool],
    seed: int = 42,
    max_attempts: int = 0,
    min_chars: int = 200,
) -> list[dict[str, Any]]:
    """Generates ``multihop`` questions verified to require the graph.

    Candidate ``(A, B, bridge)`` triples are turned into questions anchored on
    *A* but answered by the bridge chunk about *B*. A candidate is kept only when
    ``dense_miss_fn`` confirms baseline text retrieval misses the bridge chunk
    *and* ``graph_reach_fn`` confirms graph expansion reaches it -- so every kept
    question genuinely needs multi-hop retrieval.

    Args:
        graph: The knowledge graph.
        chunks: All chunks (used to build the id -> chunk lookup).
        num_questions: Target number of verified questions.
        llm: The shared in-process language model.
        dense_miss_fn: ``(question, gold_chunk_id) -> True`` if baseline dense
            retrieval does NOT return the gold chunk in its top-k.
        graph_reach_fn: ``(question, gold_chunk_id) -> True`` if graph expansion
            surfaces the gold chunk.
        seed: Sampling seed for reproducibility.
        max_attempts: Cap on candidates examined (0 = ``8 * num_questions``).
        min_chars: Minimum bridge-chunk length to be eligible.

    Returns:
        Normalised ``multihop`` rows with chunk (bridge) gold. May be shorter
        than ``num_questions`` if too few candidates pass verification.
    """
    from tqdm import tqdm

    rng = random.Random(seed)
    chunk_lookup = {c.chunk_id: c for c in chunks if len(c.text) >= min_chars}
    cap = max_attempts or (8 * num_questions)

    rows: list[dict[str, Any]] = []
    seen_gold: set[str] = set()
    attempts = 0
    bar = tqdm(total=num_questions, desc="QA synthesis [multihop]", unit="q")
    for cand in _multihop_candidates(graph, chunk_lookup, rng):
        if len(rows) >= num_questions or attempts >= cap:
            break
        attempts += 1
        chunk = cand["chunk"]
        if chunk.chunk_id in seen_gold:
            continue

        question = _ask(
            llm,
            _MULTIHOP_PROMPT.format(
                text=chunk.text, a=cand["a"], b_label=cand["b_label"]
            ),
        )
        if not question or cand["a"].lower() not in question.lower():
            continue
        if not graph_reach_fn(question, chunk.chunk_id):
            continue
        if not dense_miss_fn(question, chunk.chunk_id):
            continue

        seen_gold.add(chunk.chunk_id)
        rows.append(
            _row(
                question,
                "multihop",
                chunk.chunk_id,
                "chunk",
                chunk.doc_id,
                chunk.page_index,
            )
        )
        bar.update(1)
    bar.close()
    return rows


def normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Upgrades a legacy QA row to the typed schema in place.

    Rows written before the segmented eval carry only ``gold_chunk_id``; they are
    treated as ``text`` questions with chunk gold.

    Args:
        row: A QA row (possibly legacy).

    Returns:
        The same row, with ``qtype``, ``gold_id`` and ``gold_kind`` ensured.
    """
    row.setdefault("qtype", "text")
    if "gold_id" not in row:
        row["gold_id"] = row.get("gold_chunk_id", "")
        row["gold_kind"] = "chunk"
    return row


# Backwards-compatible alias: the original single-function entry point.
def synthesise_qa(
    chunks: list[Chunk],
    num_questions: int,
    llm: LocalLLM,
    seed: int = 42,
    min_chars: int = 200,
) -> list[dict[str, Any]]:
    """Alias for :func:`synthesise_text_qa` (preserves the original API)."""
    return synthesise_text_qa(chunks, num_questions, llm, seed, min_chars)
