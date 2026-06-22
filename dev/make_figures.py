#!/usr/bin/env python
"""Generate the architecture diagrams used in the report.

Draws two clean, paper-style figures with matplotlib (manual coordinate layout, so
boxes and arrows never overlap):

* ``figures/architecture-baseline.png`` -- the text-only baseline pipeline.
* ``figures/architecture-enhanced.png`` -- the enhanced multimodal + KG pipeline.

The third report figure, ``figures/her2-table.png``, is a static crop committed in
``figures/`` (not generated here).

Run from the repository root:  python dev/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")

# Muted, paper-style palette keyed by stage role.
_COLORS = {
    "io": ("#ECECEC", "#5A5A5A"),  # input / output
    "text": ("#D6E4F2", "#3F6FA3"),  # text retrieval
    "image": ("#D6EBDD", "#3E8159"),  # image retrieval
    "graph": ("#E6DCF2", "#6B4E9E"),  # graph retrieval
    "fuse": ("#FBE7D2", "#C07A35"),  # fusion / rerank
    "gen": ("#F6DEDE", "#B05050"),  # generation
}

_LABEL_FS = 10
_TITLE_FS = 14


class Box:
    """A positioned, styled box on the diagram (centre coordinates)."""

    def __init__(self, x, y, label, role, w=1.9, h=1.0):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label, self.role = label, role

    def draw(self, ax):
        face, edge = _COLORS[self.role]
        ax.add_patch(
            FancyBboxPatch(
                (self.x - self.w / 2, self.y - self.h / 2),
                self.w,
                self.h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6,
                mutation_aspect=1.0,
            )
        )
        ax.text(
            self.x,
            self.y,
            self.label,
            ha="center",
            va="center",
            fontsize=_LABEL_FS,
            color="#1E1E1E",
            zorder=5,
        )

    def left(self):
        return (self.x - self.w / 2, self.y)

    def right(self):
        return (self.x + self.w / 2, self.y)


def _arrow(ax, start, end):
    """Draws a single clean arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="#666666",
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=1,
        )
    )


def connect(ax, src: Box, dst: Box):
    """Connects the right edge of ``src`` to the left edge of ``dst``."""
    _arrow(ax, src.right(), dst.left())


def _new_ax(width, height, xlim, ylim, title):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=_TITLE_FS, fontweight="bold", color="#1E1E1E", pad=12)
    return fig, ax


def build_baseline():
    """Builds the baseline (text-only) architecture figure."""
    fig, ax = _new_ax(13.0, 3.0, (0, 13), (0, 3), "Baseline — text-only RAG")
    y = 1.5
    boxes = [
        Box(1.4, y, "Question", "io"),
        Box(4.0, y, "BGE-M3\ndense embed", "text"),
        Box(6.6, y, "Vector search\n(text chunks)", "text"),
        Box(9.2, y, "Top-k\ntext chunks", "text"),
        Box(11.7, y, "LLM answer\n+ citations", "gen", w=2.1),
    ]
    for b in boxes:
        b.draw(ax)
    for a, b in zip(boxes, boxes[1:]):
        connect(ax, a, b)
    _save(fig, "architecture-baseline.png")


def build_enhanced():
    """Builds the enhanced (multimodal + KG) architecture figure."""
    fig, ax = _new_ax(
        15.5,
        6.4,
        (0, 15.5),
        (0, 6.4),
        "Enhanced — multimodal + knowledge-graph RAG",
    )
    q = Box(1.6, 3.2, "Question", "io")
    # Three retrieval channels, well separated vertically.
    text_ch = Box(5.0, 5.1, "Hybrid text\n(dense + sparse)", "text", w=2.5)
    img_ch = Box(5.0, 3.2, "Image search\n(SigLIP2 → figures)", "image", w=2.5)
    graph_ch = Box(5.0, 1.3, "KG expansion\n(entity link → bridge)", "graph", w=2.5)
    fuse = Box(8.8, 3.2, "RRF fusion\n(rank-weighted)", "fuse", w=2.2)
    rerank = Box(11.6, 3.2, "Cross-encoder\nrerank", "fuse", w=2.1)
    gen = Box(14.1, 3.2, "LLM / VLM\nanswer", "gen", w=1.9)

    for b in (q, text_ch, img_ch, graph_ch, fuse, rerank, gen):
        b.draw(ax)

    for ch in (text_ch, img_ch, graph_ch):
        connect(ax, q, ch)
        connect(ax, ch, fuse)
    connect(ax, fuse, rerank)
    connect(ax, rerank, gen)

    _save(fig, "architecture-enhanced.png")


def _save(fig, name):
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    """Generates both architecture figures."""
    build_baseline()
    build_enhanced()


if __name__ == "__main__":
    main()
