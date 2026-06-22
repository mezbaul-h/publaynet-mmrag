"""End-to-end serving pipeline assembly.

:func:`build_system` wires a :class:`RAGSystem` from a configuration, loading
only the models needed at serving time. The text embedder is always loaded; the
image embedder, reranker and knowledge graph are loaded only when the
corresponding retrieval flags are set. Query-time retrieval models default to
the CPU (``models.retrieval_device``) so they do not compete with the LLM for
GPU memory now that generation runs in-process rather than via an external
server. The heavy Stage 1/2 models (OCR, captioner) are never loaded here.

A single :class:`~publaynet_mmrag.reason.llm.LocalLLM` can be injected and shared
across systems (e.g. both arms of an evaluation) to avoid loading the weights
more than once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from publaynet_mmrag.config import Config
from publaynet_mmrag.embed.image import ImageEmbedder
from publaynet_mmrag.embed.text import TextEmbedder
from publaynet_mmrag.explain.provenance import Provenance, build_provenance
from publaynet_mmrag.index.store import VectorStore
from publaynet_mmrag.kg.build import load_graph
from publaynet_mmrag.reason.generate import Generator
from publaynet_mmrag.reason.llm import LocalLLM
from publaynet_mmrag.retrieve.multimodal_rag import Retriever
from publaynet_mmrag.retrieve.rerank import Reranker
from publaynet_mmrag.types import Answer


@dataclass
class RAGSystem:
    """A fully wired retrieval-augmented generation system.

    Attributes:
        config: The active run configuration.
        retriever: The configured retriever.
        generator: The answer generator.
        llm: The shared language model.
    """

    config: Config
    retriever: Retriever
    generator: Generator
    llm: LocalLLM

    def answer(self, question: str) -> Answer:
        """Answers a question end to end.

        Args:
            question: The natural-language question.

        Returns:
            The generated :class:`~publaynet_mmrag.types.Answer`.
        """
        items, graph_paths = self.retriever.retrieve(question)
        return self.generator.generate(question, items, graph_paths)

    def explain(self, answer: Answer) -> Provenance:
        """Builds the provenance record for an answer.

        Args:
            answer: A generated answer.

        Returns:
            The :class:`~publaynet_mmrag.explain.provenance.Provenance` record.
        """
        return build_provenance(answer)


def build_llm(config: Config) -> LocalLLM:
    """Builds and loads the shared language model from configuration.

    Args:
        config: The active run configuration.

    Returns:
        A loaded :class:`~publaynet_mmrag.reason.llm.LocalLLM`.
    """
    llm = LocalLLM(
        model_name=config.models.llm_model,
        device=config.models.device,
        dtype=config.models.llm_dtype,
        load_in_4bit=config.models.llm_load_in_4bit,
        max_new_tokens=config.models.llm_max_new_tokens,
    )
    llm.load()
    return llm


def build_system(config: Config, llm: Optional[LocalLLM] = None) -> RAGSystem:
    """Builds a :class:`RAGSystem` for the given configuration.

    Only serving-time models are loaded. Enhanced components (image embedder,
    reranker, knowledge graph) are loaded lazily based on the retrieval flags.

    Args:
        config: The active run configuration.
        llm: An optional pre-loaded language model to share; one is built if not
            provided.

    Returns:
        The assembled system.
    """
    store = VectorStore(
        path=config.paths.qdrant_path,
        text_dim=config.models.text_embed_dim,
        image_dim=config.models.image_embed_dim,
    )

    retrieval_device = config.models.retrieval_device
    text_embedder = TextEmbedder(
        model_name=config.models.text_embed_model, device=retrieval_device
    )
    text_embedder.load()

    image_embedder = None
    if config.retrieval.use_image:
        image_embedder = ImageEmbedder(
            model_name=config.models.image_embed_model, device=retrieval_device
        )
        image_embedder.load()

    reranker = None
    if config.retrieval.use_rerank:
        rerank_device = config.models.rerank_device or retrieval_device
        reranker = Reranker(
            model_name=config.models.reranker_model, device=rerank_device
        )
        reranker.load()

    graph = None
    if config.retrieval.use_graph and os.path.exists(config.paths.kg_path):
        graph = load_graph(config.paths.kg_path)

    retriever = Retriever(
        config=config,
        store=store,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        reranker=reranker,
        graph=graph,
    )

    if llm is None:
        llm = build_llm(config)
    generator = Generator(llm=llm, generation=config.generation)

    return RAGSystem(config=config, retriever=retriever, generator=generator, llm=llm)
