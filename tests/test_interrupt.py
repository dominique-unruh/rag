"""Tests for the graceful (two-stage) interrupt handler."""

from __future__ import annotations

import signal

import pytest

from rag.interrupt import graceful_interrupts, interrupt_requested


def test_two_stage_interrupt(capsys: pytest.CaptureFixture[str]) -> None:
    """First SIGINT sets the flag (no raise); a second raises KeyboardInterrupt."""
    assert interrupt_requested() is False
    sentinel = object()

    def _prev(_signum: int, _frame: object) -> None:
        """Placeholder previous handler (must be restored on exit)."""

    signal.signal(signal.SIGINT, _prev)
    with graceful_interrupts():
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not _prev  # our handler is installed

        # First press: flag set, message printed, NO exception.
        handler(signal.SIGINT, None)  # type: ignore[misc]
        assert interrupt_requested() is True
        assert "Interrupt requested" in capsys.readouterr().err

        # Second press: aborts immediately.
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)  # type: ignore[misc]

    # On exit: flag cleared and the previous handler restored.
    assert interrupt_requested() is False
    assert signal.getsignal(signal.SIGINT) is _prev
    # Cleanup: restore the default handler for the rest of the suite.
    signal.signal(signal.SIGINT, signal.default_int_handler)


def test_flag_reset_on_enter() -> None:
    """Entering the context clears any stale flag from a prior run."""
    with graceful_interrupts():
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)  # type: ignore[misc]
        assert interrupt_requested() is True
    # A fresh context must start clean.
    with graceful_interrupts():
        assert interrupt_requested() is False
    signal.signal(signal.SIGINT, signal.default_int_handler)
