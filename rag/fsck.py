"""Shared fsck primitives: fix-mode prompting and problem reporting.

Kept at the top level (not under ``maintenance/``) so both the fsck tool and the
extraction plugins can use them without an import cycle (``extract`` must not
depend on ``maintenance``). The ``maintenance.fsck`` tool orchestrates checks;
this module holds the small helpers a check calls to report a problem and to
decide whether to apply its fix.
"""

from __future__ import annotations

from typing import Literal

# How the ``--fix`` flag drives a check's fixes: never fix, ask per fix, or
# always fix.
FixMode = Literal["no", "ask", "yes"]


def print_problem(message: str, *, severity: str = "error") -> None:
    """Print one problem line immediately, e.g. ``[ERROR] [email] ...``.

    ``severity`` is ``"error"`` or ``"warning"``. Checks print as they find
    problems (rather than collecting them) so a long run shows progress and so a
    fix prompt appears right under the problem it addresses.
    """
    marker = "ERROR" if severity == "error" else "warn"
    print(f"[{marker}] {message}")


def ask_fix(fix: FixMode, description: str) -> bool:
    """Decide whether to apply a fix, printing what the fix will do.

    * ``fix == "no"`` — return ``False`` without printing (report-only run).
    * ``fix == "yes"`` — print ``description`` and return ``True`` (apply it).
    * ``fix == "ask"`` — print ``description`` with a ``(y/n)`` prompt and return
      the user's answer (``True`` only on an explicit yes; EOF/anything else is no).

    ``description`` must state exactly what will change (e.g. what rows/flags), so
    an interactive user knows what they are approving.
    """
    if fix == "no":
        return False
    if fix == "yes":
        # Non-interactive apply: still announce what is being done.
        print(description)
        return True
    # fix == "ask": prompt and apply only on an explicit yes.
    try:
        answer = input(f"{description} (y/n) ").strip().lower()
    except EOFError:
        # No input available (piped/closed stdin) — treat as decline.
        return False
    return answer in ("y", "yes")
