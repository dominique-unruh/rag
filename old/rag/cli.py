"""Run an entrypoint, turning UserErrors into clean one-line messages.

Only `UserError` (and Ctrl-C) is caught — those are failures the user can fix,
carrying a message written for them. Everything else propagates with its full
traceback, because it is an internal bug we want to see. Provider/library
exceptions are converted to UserError *at their call sites* (see errors.py), not
blanket-caught here. Set RAGPROTO_DEBUG=1 to re-raise UserError with traceback too.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Callable

from .errors import UserError


def _quiet_known_warnings() -> None:
    """Silence benign, unavoidable third-party warnings from stock components."""
    # Embedded Qdrant logs this on every insert; irrelevant to local `path` mode.
    warnings.filterwarnings(
        "ignore", message="Payload indexes have no effect in the local Qdrant.*"
    )


def run(main_fn: Callable[[list[str] | None], int], argv: list[str] | None = None) -> int:
    """Invoke `main_fn(argv)`, mapping UserError / Ctrl-C to a clean exit code."""
    _quiet_known_warnings()
    if os.environ.get("RAGPROTO_DEBUG"):
        return main_fn(argv)
    try:
        return main_fn(argv)
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 130
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
