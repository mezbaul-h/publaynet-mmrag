#!/usr/bin/env python
"""Interactive Gradio demo.

Presents a query box and a baseline/enhanced toggle, then shows the generated
answer, the per-evidence provenance table, a gallery of the retrieved
figure/table crops (cited ones flagged, with their captions), and the
knowledge-graph paths used. Crops are read back from the Stage 1 artifacts on
disk (``region.crop_path``).
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


def _ensure_demo_deps() -> None:
    """Checks the optional demo dependency is installed.

    Raises:
        ImportError: With install guidance if Gradio is missing.
    """
    try:
        import gradio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The demo needs the optional 'demo' extra (Gradio):\n"
            '    pip install -e ".[demo]"'
        ) from exc


_ensure_demo_deps()

import gradio as gr  # noqa: E402

from publaynet_mmrag.config import Config, load_config  # noqa: E402
from publaynet_mmrag.pipeline import RAGSystem, build_system  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import Region, read_jsonl  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")
_SYSTEMS: dict[str, RAGSystem] = {}
_REGION_CACHE: dict[str, Region] = {}


def _resolve_crop(crop_path: str) -> str | None:
    """Resolves a stored (possibly repo-relative) crop path to an existing file.

    Args:
        crop_path: The ``region.crop_path`` recorded at ingest time.

    Returns:
        An absolute path to the crop if it exists, else ``None``.
    """
    if not crop_path:
        return None
    candidate = (
        crop_path if os.path.isabs(crop_path) else os.path.join(_REPO_ROOT, crop_path)
    )
    return candidate if os.path.exists(candidate) else None


def _load_config(mode: str) -> Config:
    """Loads the configuration for a mode.

    Args:
        mode: ``"baseline"`` or ``"enhanced"``.

    Returns:
        The composed configuration.
    """
    base = os.path.join(_CONFIG_DIR, "base.yaml")
    config = load_config(base, os.path.join(_CONFIG_DIR, f"{mode}.yaml"))
    config.mode = mode
    # Answer with a VLM that reads the retrieved figure/table crops, so visual
    # questions are answered from the image rather than a lossy caption.
    config.generation.vision_generation = True
    return config


def _get_system(mode: str) -> RAGSystem:
    """Returns a cached system for the mode, building it on first use.

    Args:
        mode: ``"baseline"`` or ``"enhanced"``.

    Returns:
        The :class:`RAGSystem` for that mode.
    """
    if mode not in _SYSTEMS:
        _SYSTEMS[mode] = build_system(_load_config(mode))
    return _SYSTEMS[mode]


def _index_regions(config: Config) -> None:
    """Loads all regions into a cache keyed by region id.

    Args:
        config: The active configuration (for the regions directory).
    """
    if _REGION_CACHE:
        return
    for path in glob.glob(os.path.join(config.paths.regions_dir, "*.jsonl")):
        for row in read_jsonl(path):
            region = Region.from_dict(row)
            _REGION_CACHE[region.region_id] = region


def answer_query(question: str, mode: str):
    """Answers a question and assembles the demo outputs.

    Args:
        question: The user's question.
        mode: The selected pipeline arm.

    Returns:
        A tuple of (answer markdown, evidence rows, figure gallery, graph-paths
        markdown). The gallery holds ``(crop_path, caption)`` pairs for the
        retrieved figure/table regions.
    """
    import time

    start = time.perf_counter()
    system = _get_system(mode)
    _index_regions(system.config)
    answer = system.answer(question)
    provenance = system.explain(answer)
    elapsed = format_duration(time.perf_counter() - start)
    print(f"[demo] answered ({mode}) in {elapsed}")

    md = f"### Answer\n{answer.text}\n"
    md += f"\n_Answered in {elapsed}._"
    if answer.reasoning:
        md += (
            f"\n<details><summary>Reasoning</summary>\n\n{answer.reasoning}\n</details>"
        )

    rows = [
        [
            t.source_id,
            t.doc_id,
            t.page_index,
            t.modality,
            t.retrieval_source,
            round(t.score, 3),
            "yes" if t.cited else "",
        ]
        for t in provenance.evidence
    ]

    # Gallery of retrieved figure/table crops; cited ones are flagged first.
    gallery: list[tuple[str, str]] = []
    for t in provenance.evidence:
        if t.modality != "image":
            continue
        crop = _resolve_crop(t.crop_path)
        if not crop:
            continue
        caption = (t.text or "").strip() or "(no caption)"
        flag = "★ CITED — " if t.cited else ""
        label = f"{flag}{t.doc_id} p{t.page_index} · {caption}"
        gallery.append((crop, label[:200]))
    gallery.sort(key=lambda item: not item[1].startswith("★"))

    paths_md = (
        "### Knowledge-graph paths\n"
        + "\n".join(f"- {p}" for p in provenance.graph_paths)
        if provenance.graph_paths
        else "_No graph paths used._"
    )
    return md, rows, gallery, paths_md


def build_ui() -> "gr.Blocks":
    """Builds the Gradio interface.

    Returns:
        The assembled Gradio ``Blocks`` app.
    """
    with gr.Blocks(title="PubLayNet Multimodal RAG") as demo:
        gr.Markdown(
            "# PubLayNet Multimodal RAG\nAsk a question over the indexed pages."
        )
        with gr.Row():
            question = gr.Textbox(label="Question", scale=4)
            mode = gr.Radio(
                ["baseline", "enhanced"], value="enhanced", label="Pipeline", scale=1
            )
        run_button = gr.Button("Ask", variant="primary")
        answer_md = gr.Markdown()
        evidence = gr.Dataframe(
            headers=["source", "doc", "page", "modality", "channel", "score", "cited"],
            label="Evidence",
            wrap=True,
        )
        figures = gr.Gallery(
            label="Retrieved figures / tables (★ = cited)",
            columns=3,
            height="auto",
            object_fit="contain",
        )
        paths_md = gr.Markdown()

        run_button.click(
            answer_query,
            inputs=[question, mode],
            outputs=[answer_md, evidence, figures, paths_md],
        )
    return demo


def main() -> None:
    """Launches the demo."""
    build_ui().launch()


if __name__ == "__main__":
    main()
