#!/usr/bin/env python
"""Stage 5: serve the pipeline over HTTP with FastAPI.

Exposes a small JSON API around a single :class:`RAGSystem`. The system is built
once at startup from the selected configuration (baseline or enhanced) so model
loading is paid only on boot. Run with uvicorn; see the README for the command.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


def _ensure_serve_deps() -> None:
    """Checks the optional serving dependencies are installed.

    Raises:
        ImportError: With install guidance if FastAPI/uvicorn are missing.
    """
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Serving needs the optional 'serve' extra (FastAPI + uvicorn):\n"
            '    pip install -e ".[serve]"'
        ) from exc


_ensure_serve_deps()

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from publaynet_mmrag.pipeline import RAGSystem, build_system  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from scripts._common import resolve_config  # noqa: E402

app = FastAPI(title="PubLayNet Multimodal RAG")
_system: RAGSystem | None = None


class QueryRequest(BaseModel):
    """Request body for the query endpoints.

    Attributes:
        question: The natural-language question to answer.
    """

    question: str


def get_system() -> RAGSystem:
    """Returns the lazily constructed RAG system.

    Returns:
        The process-wide :class:`RAGSystem`.

    Raises:
        RuntimeError: If the system has not been initialised.
    """
    if _system is None:
        raise RuntimeError("System not initialised. Start via the module main().")
    return _system


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """Answers a question and returns the answer with citations.

    Args:
        request: The query request.

    Returns:
        A dictionary with the answer text, reasoning and citations.
    """
    answer = get_system().answer(request.question)
    return {
        "question": answer.question,
        "answer": answer.text,
        "reasoning": answer.reasoning,
        "citations": answer.citations,
        "graph_paths": answer.graph_paths,
    }


@app.post("/explain")
def explain(request: QueryRequest) -> dict:
    """Answers a question and returns the full provenance trace.

    Args:
        request: The query request.

    Returns:
        The serialised provenance record.
    """
    system = get_system()
    answer = system.answer(request.question)
    return system.explain(answer).to_dict()


def main() -> None:
    """Builds the system and starts the HTTP server."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Stage 5: serve the pipeline.")
    parser.add_argument("--config", required=False)
    parser.add_argument("--mode", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), os.pardir, "configs", "base.yaml"
        )

    global _system
    import time

    start = time.perf_counter()
    _system = build_system(resolve_config(args))
    print(
        f"System ready in {format_duration(time.perf_counter() - start)}; "
        f"serving on http://{args.host}:{args.port}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
