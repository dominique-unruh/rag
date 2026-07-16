"""Allow running translation via ``python -m rag.translate``.

Delegates to the shared top-level CLI, injecting the ``translate`` subcommand so
``python -m rag.translate --limit 10`` behaves like ``rag translate --limit 10``
(PLAN §8).
"""

from __future__ import annotations

import sys

from ..cli import main

if __name__ == "__main__":
    # Prepend the subcommand name so the shared parser routes to translation.
    raise SystemExit(main(["translate", *sys.argv[1:]]))
