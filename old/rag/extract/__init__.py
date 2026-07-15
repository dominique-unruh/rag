"""Extraction pipeline (sources -> Document Store).

    python -m rag.extract [--only email|notes] [--verbose]
"""

from __future__ import annotations

import argparse
import sys

from .. import cli, config, runtime, stores
from .base import Extractor, run_extractor
from .email import EmailExtractor
from .notes import NotesExtractor

EXTRACTORS: dict[str, Extractor] = {
    NotesExtractor.name: NotesExtractor(),
    EmailExtractor.name: EmailExtractor(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.extract", description="Extract sources into the Document Store."
    )
    parser.add_argument(
        "--only", choices=sorted(EXTRACTORS), help="run a single extractor."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def run_extract(cfg, only: str | None = None, verbose: bool = False) -> None:
    names = [only] if only else list(EXTRACTORS)
    store = stores.open_document_store(cfg)
    try:
        for name in names:
            ex = EXTRACTORS[name]
            if not ex.enabled(cfg):
                print(f"{name}: disabled, skipped")
                continue
            run_extractor(ex, cfg, store, verbose=verbose)
        print(f"document store: {store.count()} documents total")
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load(".")
    runtime.configure(cfg)  # embeddings/LLM unused here, but keeps Settings consistent
    run_extract(cfg, only=args.only, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
