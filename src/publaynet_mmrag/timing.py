"""Small timing utilities for the stage scripts."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds as a compact human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string such as ``"7.4s"``, ``"3m 05s"`` or ``"1h 02m 03s"``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"
