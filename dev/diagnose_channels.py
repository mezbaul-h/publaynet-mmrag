#!/usr/bin/env python
"""Diagnostic: are the image and graph retrieval channels actually firing?

The ablation showed the image and graph channels with exactly zero effect on
retrieval metrics. This distinguishes the two possible causes:

  * the channel *fires* (returns candidates) but they do not change the top-k
    ranking -- e.g. image regions can never match a text-chunk gold, which is
    structural to the text-based QA evaluation; versus
  * the channel returns *nothing* (silently broken: empty image collection,
    or graph entity-linking matching no query terms).

It builds only the retrieval pieces it needs (no LLM, no reranker), reports the
index and graph sizes, then runs the real eval questions through the image and
graph channels directly, reporting how often each returns candidates.

Run from the repository root:
    python dev/diagnose_channels.py            # 30 sample questions
    python dev/diagnose_channels.py 100        # 100 sample questions
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from publaynet_mmrag.config import load_config  # noqa: E402
from publaynet_mmrag.embed.image import ImageEmbedder  # noqa: E402
from publaynet_mmrag.index import schema  # noqa: E402
from publaynet_mmrag.index.store import VectorStore  # noqa: E402
from publaynet_mmrag.kg import query as kg_query  # noqa: E402
from publaynet_mmrag.kg.build import CHUNK, ENTITY, load_graph  # noqa: E402
from publaynet_mmrag.types import read_jsonl  # noqa: E402

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "configs")


def _count(store: VectorStore, collection: str) -> int:
    """Returns the number of points in a collection (0 if absent)."""
    try:
        if not store.client.collection_exists(collection):
            return 0
        return store.client.count(collection_name=collection, exact=True).count
    except Exception:
        return -1


def main() -> None:
    """Runs the channel-firing diagnostic."""
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    config = load_config(
        os.path.join(_CONFIG_DIR, "base.yaml"),
        os.path.join(_CONFIG_DIR, "enhanced.yaml"),
    )

    store = VectorStore(
        path=config.paths.qdrant_path,
        text_dim=config.models.text_embed_dim,
        image_dim=config.models.image_embed_dim,
    )
    n_text = _count(store, schema.TEXT_COLLECTION)
    n_image = _count(store, schema.IMAGE_COLLECTION)
    print(f"Index: {n_text} text points, {n_image} image points")

    graph = None
    if os.path.exists(config.paths.kg_path):
        graph = load_graph(config.paths.kg_path)
        entities = sum(1 for _, d in graph.nodes(data=True) if d.get("ntype") == ENTITY)
        chunks = sum(1 for _, d in graph.nodes(data=True) if d.get("ntype") == CHUNK)
        print(
            f"Graph: {graph.number_of_nodes()} nodes "
            f"({entities} entities, {chunks} chunks), "
            f"{graph.number_of_edges()} edges"
        )
    else:
        print("Graph: none found at", config.paths.kg_path)

    image_embedder = ImageEmbedder(
        model_name=config.models.image_embed_model,
        device=config.models.retrieval_device,
    )
    image_embedder.load()

    qa = read_jsonl(config.paths.qa_path)
    sample = qa[:sample_size]
    n = len(sample)
    if n == 0:
        print("No questions found in", config.paths.qa_path)
        return

    image_fired = 0
    image_hits = 0
    graph_fired = 0
    graph_docs = 0
    graph_paths = 0
    for row in sample:
        question = row["question"]

        vector = image_embedder.embed_text([question])[0]
        hits = store.search_image(vector, limit=config.retrieval.top_k)
        if hits:
            image_fired += 1
        image_hits += len(hits)

        if graph is not None:
            expansion = kg_query.expand(
                graph, question, hops=config.retrieval.graph_hops
            )
            if expansion.doc_ids or expansion.paths:
                graph_fired += 1
            graph_docs += len(expansion.doc_ids)
            graph_paths += len(expansion.paths)

    print(f"\nOver {n} sample questions:")
    print(
        f"  Image: returned >=1 region on {image_fired}/{n} queries; "
        f"{image_hits} hits total (avg {image_hits / n:.1f}/query)."
    )
    print(
        f"  Graph: expanded on {graph_fired}/{n} queries; "
        f"{graph_docs} doc_ids, {graph_paths} paths total."
    )
    print(
        "\nReading:\n"
        "  - Image hits > 0 but zero retrieval delta => the channel fires, but a\n"
        "    figure region cannot match a text-chunk gold (structural to this eval).\n"
        "  - Graph doc_ids > 0 but zero delta => neighbours are added but never\n"
        "    outrank the gold; doc_ids == 0 everywhere => entity linking matched\n"
        "    nothing (broken or too-sparse graph).\n"
        "  - Either count == 0 across all queries => that channel is not firing."
    )


if __name__ == "__main__":
    main()
