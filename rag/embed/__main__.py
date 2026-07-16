"""Allow running embedding via ``python -m rag.embed``.

Delegates to the shared top-level CLI, injecting the ``embed`` subcommand so
``python -m rag.embed --model qwen3-8b`` behaves like ``rag embed --model
qwen3-8b`` (PLAN §8).
"""

from __future__ import annotations

import sys

from ..cli import main

if __name__ == "__main__":
    # Prepend the subcommand name so the shared parser routes to embedding.
    raise SystemExit(main(["embed", *sys.argv[1:]]))
