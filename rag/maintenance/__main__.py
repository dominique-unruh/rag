"""Allow running maintenance tools via ``python -m rag.maintenance``.

Delegates to the shared top-level CLI, injecting the ``maintenance`` subcommand
so ``python -m rag.maintenance db-browser`` behaves like ``rag maintenance
db-browser`` (PLAN §8).
"""

from __future__ import annotations

import sys

from ..cli import main

if __name__ == "__main__":
    # Prepend the subcommand name so the shared parser routes to maintenance.
    raise SystemExit(main(["maintenance", *sys.argv[1:]]))
