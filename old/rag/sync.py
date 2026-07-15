"""Run all three pipelines in order: extract -> embed -> graph.

    python -m rag.sync [--verbose] [--delay SECONDS] [--only email|notes]

embed and graph are independent of each other; both depend on extract.
"""

from __future__ import annotations

import argparse
import sys

from . import cli, config, runtime
from .embed import run_embed
from .extract import run_extract
from .graph import run_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag.sync", description="Extract, embed, graph.")
    parser.add_argument("--only", choices=("email", "notes"),
                        help="restrict extraction to one source.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    delay = max(0.0, args.delay)
    cfg = config.load(".")
    runtime.configure(cfg)
    runtime.verify_embedding_dim(cfg)
    print("== extract =="); run_extract(cfg, only=args.only, verbose=args.verbose)
    print("== embed ==");   run_embed(cfg, verbose=args.verbose, delay=delay)
    print("== graph ==");   run_graph(cfg, verbose=args.verbose, delay=delay)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
