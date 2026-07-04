"""Embedding pipeline (Document Store -> Qdrant).

Uniform for every extractor. Diffs the Document Store against the `embed_state`
ledger and converges Qdrant: embed new/changed documents, delete vectors of
removed/changed ones. Per-document commit = crash-safe / resumable.

    python -m rag.embed [--verbose] [--delay SECONDS]
"""

from __future__ import annotations

import argparse
import sys
import time

from llama_index.core import StorageContext, VectorStoreIndex

from .. import cli, config, errors, runtime, stores
from ..documents import filter_changes, plan_changes, to_document
from ..htmllog import HtmlLog, source_label

LEDGER = "embed_state"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.embed", description="Embed Document-Store documents into Qdrant."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                        help="pause between documents to throttle the endpoint.")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="process at most N new/changed documents this run.")
    parser.add_argument("--docs", default=None, metavar="PATTERN",
                        help="glob (*,?) on doc_id; only matching documents are processed.")
    return parser


def run_embed(cfg, verbose: bool = False, delay: float = 0.0,
              max_items: int | None = None, docs_pattern: str | None = None) -> None:
    store = stores.open_document_store(cfg)
    client = stores.open_qdrant_client(cfg)
    log = HtmlLog(stores.logs_dir(cfg) / "embed.html", "rag embed log")
    try:
        vector_store = stores.build_vector_store(client, cfg)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=runtime.Settings.embed_model,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
        )

        want = store.want_hashes()
        have = store.state(LEDGER)
        to_process, to_delete = plan_changes(want, have)
        to_process, to_delete = filter_changes(to_process, to_delete, docs_pattern, max_items)
        print(f"embed: {len(to_process)} to embed, {len(to_delete)} to delete "
              f"({len(want)} documents)")

        for doc_id in to_delete:
            vector_store.delete(doc_id)
            store.unmark(LEDGER, doc_id)
            if verbose:
                print(f"  delete {doc_id}")

        for i, doc_id in enumerate(to_process, 1):
            if doc_id in have:  # changed -> clear old vectors first
                vector_store.delete(doc_id)
            sdoc = store.get(doc_id)
            if verbose:
                print(f"  embed {i}/{len(to_process)}: {doc_id}")
            with errors.endpoint_errors(f"({cfg.models.api_base})"):
                index.insert(to_document(sdoc))
            store.mark(LEDGER, doc_id, sdoc.doc_hash)
            log.add(doc_id, source_label(sdoc), sdoc.text)
            if delay and i < len(to_process):
                time.sleep(delay)
        print("done.")
    finally:
        store.close()
        close = getattr(client, "close", None)
        if callable(close):
            close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load(".")
    runtime.configure(cfg)
    runtime.verify_embedding_dim(cfg)
    run_embed(cfg, verbose=args.verbose, delay=max(0.0, args.delay),
              max_items=args.max, docs_pattern=args.docs)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
