"""Suppression of third-party progress bars.

Libraries such as FlagEmbedding (BGE-M3, the reranker) and GLiNER print their
own per-call ``tqdm`` bars to stderr. During a stage these scroll the
pipeline's own progress bar out of view. :func:`silence_stderr` redirects
stderr around those calls so only the stage bar remains.

This is safe because a stage's ``tqdm`` bar captures the real stderr when it is
constructed (before any suppression), so it keeps rendering; and redirecting
stderr only hides *written* text -- exceptions still propagate normally.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator


@contextlib.contextmanager
def silence_stderr() -> Iterator[None]:
    """Temporarily redirects stderr to the null device.

    Yields:
        Control with stderr suppressed; stderr is always restored on exit.
    """
    devnull = open(os.devnull, "w")
    saved = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = saved
        devnull.close()
