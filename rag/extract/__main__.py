"""Allow running extraction directly via ``python -m rag.extract``.

Delegates to the shared top-level CLI, injecting the ``extract`` subcommand so
``python -m rag.extract email count=100`` behaves like ``rag extract email
count=100`` (PLAN §8).
"""

from __future__ import annotations

import sys

from ..cli import main

if __name__ == "__main__":
    # Prepend the subcommand name so the shared parser routes to extraction.
    raise SystemExit(main(["extract", *sys.argv[1:]]))
