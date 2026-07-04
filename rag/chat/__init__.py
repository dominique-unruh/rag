"""Interactive chat over the ingested corpus.

Uses the stock PropertyGraphIndex chat engine in `condense_plus_context` mode,
which condenses the running conversation into a standalone question, retrieves
context via the index's default retrievers (VectorContextRetriever over Qdrant +
LLMSynonymRetriever over the graph), and answers with that context in the prompt.

Run:  python -m rag.chat
"""

from __future__ import annotations

import argparse
import sys

from .. import cli, config, errors, runtime, stores
from . import trace

BANNER = (
    "rag chat — ask questions over your notes/email.\n"
    "Commands: /reset (clear history), /exit or Ctrl-D to quit.\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.chat", description="Chat over the ingested corpus."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print every store query + result (vector/graph), not model thinking.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = config.load(".")
    runtime.configure(cfg)
    runtime.verify_embedding_dim(cfg)

    client = stores.open_qdrant_client(cfg)
    graph_store, graph_db = stores.open_graph_store(cfg)
    try:
        vector_store = stores.build_vector_store(client, cfg)
        index = stores.open_read_index(cfg, graph_store, vector_store)

        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=runtime.Settings.llm,
        )
        if args.trace:
            n = trace.enable(chat_engine._retriever)
            print(f"(tracing {n} store retriever(s))", file=sys.stderr)

        print(BANNER)
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user in ("/exit", "/quit"):
                break
            if user == "/reset":
                chat_engine.reset()
                print("(history cleared)")
                continue

            # A transient endpoint error ends the turn, not the whole session.
            try:
                with errors.endpoint_errors(f"({cfg.models.api_base})"):
                    response = chat_engine.chat(user)
            except errors.UserError as exc:
                print(f"error: {exc}\n", file=sys.stderr)
                continue
            print(f"\nbot> {response}\n")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        stores.close_graph_db(graph_db)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
