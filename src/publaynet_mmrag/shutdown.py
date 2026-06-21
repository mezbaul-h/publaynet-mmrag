"""Graceful shutdown on Ctrl-C / SIGTERM.

Long stages are frequently interrupted. :func:`graceful_shutdown` turns a
``KeyboardInterrupt`` (Ctrl-C) or ``SIGTERM`` (``kill``) into a clean exit: an
optional cleanup callback runs first (e.g. saving the knowledge graph or
regenerating the chunk file from completed pages), a short message is printed
instead of a traceback, and the process exits with the conventional code 130.

``SIGKILL`` (``kill -9``) cannot be caught; nothing can run on that path. The
stages are resumable regardless, so a re-run continues from the last persisted
state in every case.
"""

from __future__ import annotations

import contextlib
import signal
import sys
from collections.abc import Callable, Iterator


@contextlib.contextmanager
def graceful_shutdown(
    on_interrupt: Callable[[], None] | None = None,
    message: str = "Interrupted.",
) -> Iterator[None]:
    """Runs a block, exiting cleanly on Ctrl-C or SIGTERM.

    Args:
        on_interrupt: Optional cleanup to run before exiting (e.g. save
            progress). Exceptions raised by it are reported, not propagated.
        message: Short message printed on interruption.

    Yields:
        Control to the wrapped block.
    """

    def _raise_keyboard_interrupt(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        yield
    except KeyboardInterrupt:
        print(f"\n{message}", file=sys.stderr)
        if on_interrupt is not None:
            try:
                on_interrupt()
            except Exception as exc:  # pragma: no cover
                print(f"Cleanup during shutdown failed: {exc}", file=sys.stderr)
        sys.exit(130)
    finally:
        signal.signal(signal.SIGTERM, previous)
