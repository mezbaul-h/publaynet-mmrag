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
from publaynet_mmrag.types import Chunk, read_jsonl  # noqa: E402
from scripts._common import add_config_args, resolve_config  # noqa: E402


def run(config: Config) -> None:
    """Runs Stage 3 knowledge-graph construction.

    Args:
        config: The active run configuration.
    """
    chunks = [Chunk.from_dict(row) for row in read_jsonl(config.paths.chunks_path)]

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

    builder = KnowledgeGraphBuilder(cooccurrence=config.kg.cooccurrence)
    from tqdm import tqdm

    for chunk in tqdm(chunks, desc="Stage 3: KG", unit="chunk"):
        entities = entity_extractor.extract(chunk.text)
        triples = relation_extractor.extract(chunk.text) if relation_extractor else []
        builder.add_chunk(chunk, entities, triples)

    entity_extractor.unload()
    if llm is not None:
        llm.unload()
    builder.save(config.paths.kg_path)
    print(
        f"Stage 3 complete: {builder.graph.number_of_nodes()} nodes, "
        f"{builder.graph.number_of_edges()} edges -> {config.paths.kg_path}"
    )


def main() -> None:
    """Parses arguments and runs Stage 3."""
    parser = argparse.ArgumentParser(description="Stage 3: build the knowledge graph.")
    add_config_args(parser)
    args = parser.parse_args()
    run(resolve_config(args))


if __name__ == "__main__":
    main()
