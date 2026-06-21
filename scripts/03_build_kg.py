#!/usr/bin/env python
"""Stage 3: build the knowledge graph.

Loads the Stage 1 chunks, extracts entities with GLiNER and (optionally)
relation triples with the local LLM, builds a NetworkX multigraph linking
documents, chunks and entities, and writes it to GraphML. GLiNER is loaded and
unloaded around the pass so the GPU is free afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from publaynet_mmrag.config import Config  # noqa: E402
from publaynet_mmrag.kg.build import KnowledgeGraphBuilder  # noqa: E402
from publaynet_mmrag.kg.extract import (  # noqa: E402
    EntityExtractor,
    RelationExtractor,
)
from publaynet_mmrag.reason.llm import LocalLLM  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import Chunk, read_jsonl  # noqa: E402
from scripts._common import add_config_args, resolve_config  # noqa: E402


def run(config: Config) -> None:
    """Runs Stage 3 knowledge-graph construction.

    Resumable: if a graph already exists it is loaded and the chunks already in
    it are skipped, so a re-run continues rather than rebuilding. The graph is
    checkpointed to disk every ``checkpoint_every`` chunks (atomically), so a
    crash loses at most that many chunks of work. Models are loaded lazily, so a
    fully-completed resume loads nothing. To rebuild from scratch, delete the
    graph file.

    Args:
        config: The active run configuration.
    """
    chunks = [Chunk.from_dict(row) for row in read_jsonl(config.paths.chunks_path)]
    os.makedirs(os.path.dirname(config.paths.kg_path) or ".", exist_ok=True)

    import time

    start = time.perf_counter()

    # Resume from an existing graph if present.
    if os.path.exists(config.paths.kg_path):
        from publaynet_mmrag.kg.build import load_graph

        builder = KnowledgeGraphBuilder.from_graph(
            load_graph(config.paths.kg_path), cooccurrence=config.kg.cooccurrence
        )
        done = builder.processed_chunk_ids()
    else:
        builder = KnowledgeGraphBuilder(cooccurrence=config.kg.cooccurrence)
        done = set()

    pending = [c for c in chunks if c.chunk_id not in done]
    if done:
        print(
            f"Resuming KG: {len(done)} chunks already in graph, {len(pending)} to go."
        )
    if not pending:
        print("All chunks already in the graph; nothing to do.")
        builder.save(config.paths.kg_path)
        print(f"Stage 3 finished in {format_duration(time.perf_counter() - start)}.")
        return

    entity_extractor = EntityExtractor(
        model_name=config.models.ner_model,
        labels=config.kg.entity_labels,
        threshold=config.kg.ner_threshold,
    )
    entity_extractor.load()

    relation_extractor = None
    llm = None
    if config.kg.use_llm_relations:
        llm = LocalLLM(
            model_name=config.models.llm_model,
            device=config.models.device,
            dtype=config.models.llm_dtype,
            load_in_4bit=config.models.llm_load_in_4bit,
            max_new_tokens=config.models.llm_max_new_tokens,
        )
        llm.load()
        relation_extractor = RelationExtractor(llm=llm)

    from tqdm import tqdm

    checkpoint_every = config.kg.checkpoint_every
    for i, chunk in enumerate(tqdm(pending, desc="Stage 3: KG", unit="chunk"), start=1):
        entities = entity_extractor.extract(chunk.text)
        triples = relation_extractor.extract(chunk.text) if relation_extractor else []
        builder.add_chunk(chunk, entities, triples)
        if checkpoint_every and i % checkpoint_every == 0:
            builder.save(config.paths.kg_path)

    entity_extractor.unload()
    if llm is not None:
        llm.unload()
    builder.save(config.paths.kg_path)
    print(
        f"Stage 3 complete: {builder.graph.number_of_nodes()} nodes, "
        f"{builder.graph.number_of_edges()} edges -> {config.paths.kg_path}"
    )
    print(f"Stage 3 finished in {format_duration(time.perf_counter() - start)}.")


def main() -> None:
    """Parses arguments and runs Stage 3."""
    parser = argparse.ArgumentParser(description="Stage 3: build the knowledge graph.")
    add_config_args(parser)
    args = parser.parse_args()
    run(resolve_config(args))


if __name__ == "__main__":
    main()
