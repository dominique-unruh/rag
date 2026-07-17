"""Cooperative (graceful) interrupt handling for long-running pipeline runs.

The first Ctrl-C does not abort immediately: it sets a global flag and prints a
notice. Pipeline loops call :func:`interrupt_requested` at item boundaries
("checkpoints" — after one item extracted / translated / embedded) and stop
cleanly, so partial work is committed rather than lost. A **second** Ctrl-C
restores the default handler and raises :class:`KeyboardInterrupt` immediately,
so the user can always force an abort.

Only the main thread can install signal handlers; the pipeline CLI runs there.
"""

from __future__ import annotations

import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

# Module-global flag: set once the user has requested a graceful interrupt. Read
# by pipeline loops via :func:`interrupt_requested`.
_interrupt_requested: bool = False


def interrupt_requested() -> bool:
    """Return whether a graceful interrupt has been requested (checkpoint test)."""
    return _interrupt_requested


@contextmanager
def graceful_interrupts() -> Iterator[None]:
    """Install a two-stage SIGINT handler for the duration of the ``with`` block.

    First Ctrl-C: set the global flag and print a notice (the run continues to its
    next checkpoint). Second Ctrl-C: restore the previous handler and raise
    :class:`KeyboardInterrupt` at once. The previous handler and the flag are both
    restored/cleared on exit, so the context is reusable and nesting-safe.
    """
    global _interrupt_requested
    # Start from a clean slate so a prior run's flag never leaks into this one.
    _interrupt_requested = False
    previous = signal.getsignal(signal.SIGINT)

    def _handler(signum: int, frame: FrameType | None) -> None:
        """SIGINT handler: request-then-abort across two presses."""
        global _interrupt_requested
        if _interrupt_requested:
            # Second press: hand control back to the default behaviour and abort now.
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        # First press: note the request; loops stop at their next checkpoint.
        _interrupt_requested = True
        print(
            "Interrupt requested; stopping after the current item "
            "(press Ctrl-C again to abort immediately).",
            file=sys.stderr,
            flush=True,
        )

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        # Always restore the prior handler and clear the flag on the way out.
        signal.signal(signal.SIGINT, previous)
        _interrupt_requested = False
