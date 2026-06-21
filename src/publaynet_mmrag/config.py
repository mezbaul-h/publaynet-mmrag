"""Typed configuration loaded from composable YAML files.

A run is configured by overlaying a variant file (``baseline.yaml`` or
``enhanced.yaml``) on top of ``base.yaml``. The baseline disables every
enhanced component (sparse retrieval, image modality, knowledge graph,
reranking) so that the only variables between the two pipelines are modality
and structured knowledge -- which is what the quantitative comparison isolates.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Filesystem locations for inputs, caches and stage artifacts."""

    data_root: str = "data"
    crops_dir: str = "data/crops"
    regions_dir: str = "data/regions"
    chunks_path: str = "data/chunks.jsonl"
    qdrant_path: str = "data/qdrant"
    kg_path: str = "data/kg.graphml"
    qa_path: str = "data/qa.jsonl"
    results_dir: str = "data/results"


class IngestConfig(BaseModel):
    """Dataset ingestion settings."""

    source: str = Field(
        default="webdataset",
        description="Ingestion backend: 'webdataset' (subset) or 'coco' (full).",
    )
    shard_urls: list[str] = Field(
        default_factory=lambda: [
            "https://huggingface.co/datasets/lhoestq/small-publaynet-wds/"
            "resolve/main/publaynet-train-00000%d.tar" % i
            for i in range(4)
        ]
    )
    coco_annotations: str = ""
    coco_image_dir: str = ""
    max_pages: int = 0  # 0 means no limit.
    streaming: bool = True  # WebDataset only; False downloads shards up front.


class ModelsConfig(BaseModel):
    """Model identifiers and the devices each stage runs on."""

    device: str = "cuda"  # Index/build stages (throughput-bound).
    retrieval_device: str = "cpu"  # Serve-time query models, to spare VRAM.
    text_embed_model: str = "BAAI/bge-m3"
    text_embed_dim: int = 1024
    image_embed_model: str = "google/siglip2-base-patch16-224"
    image_embed_dim: int = 768
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ner_model: str = "urchade/gliner_medium-v2.1"
    caption_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"  # Hugging Face model id.
    llm_dtype: str = "float16"
    llm_load_in_4bit: bool = False  # Requires the optional 'quant' extra.
    llm_max_new_tokens: int = 512


class ChunkConfig(BaseModel):
    """Text chunking parameters."""

    max_chars: int = 1200
    overlap_chars: int = 150
    min_chars: int = 40


class RetrievalConfig(BaseModel):
    """Retrieval behaviour; the enhanced toggles are the experimental knobs."""

    top_k: int = 5
    candidate_k: int = 30
    use_sparse: bool = False
    use_image: bool = False
    use_graph: bool = False
    use_rerank: bool = False
    graph_hops: int = 1
    image_weight: float = 0.5


class KGConfig(BaseModel):
    """Knowledge-graph construction settings."""

    entity_labels: list[str] = Field(
        default_factory=lambda: [
            "method",
            "dataset",
            "metric",
            "result",
            "author",
            "task",
            "model",
        ]
    )
    ner_threshold: float = 0.5
    use_llm_relations: bool = True
    cooccurrence: bool = True


class GenerationConfig(BaseModel):
    """Answer-generation settings."""

    temperature: float = 0.1
    max_context_items: int = 6
    chain_of_thought: bool = True
    num_ctx: int = 8192


class EvalConfig(BaseModel):
    """Evaluation harness settings."""

    num_questions: int = 300
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    use_llm_judge: bool = True
    judge_sample_size: int = 150  # LLM-judge only this many; 0 = all questions.
    seed: int = 42


class Config(BaseModel):
    """Top-level configuration for a single pipeline run."""

    mode: str = "baseline"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    kg: KGConfig = Field(default_factory=KGConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges ``overlay`` into a copy of ``base``.

    Args:
        base: The base mapping.
        overlay: The mapping whose values take precedence.

    Returns:
        A new merged dictionary; inputs are not mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(base_path: str, variant_path: str | None = None) -> Config:
    """Loads and validates a configuration from one or two YAML files.

    Args:
        base_path: Path to the base YAML configuration.
        variant_path: Optional path to a variant YAML overlaid on the base.

    Returns:
        The validated :class:`Config`.
    """
    with open(base_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if variant_path:
        with open(variant_path, "r", encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        data = _deep_merge(data, overlay)
    return Config.model_validate(data)
