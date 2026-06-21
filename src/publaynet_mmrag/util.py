"""Small shared utilities (timing and human-readable durations)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds as a compact human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string such as ``"5s"``, ``"1m 5s"`` or ``"2h 14m 33s"``. Units are
        shown from the largest non-zero unit down to seconds (seconds always).
    """
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Times a block and prints how long it took on successful completion.

    Args:
        label: A label for the finished step (e.g. ``"Stage 1"``).

    Yields:
        Control to the wrapped block. On normal exit, prints
        ``"{label} finished in {duration}."``; if the block raises, nothing is
        printed and the exception propagates.
    """
    start = time.perf_counter()
    yield
    print(f"{label} finished in {format_duration(time.perf_counter() - start)}.")
