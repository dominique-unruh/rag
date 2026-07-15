"""Property-graph extraction pipeline (Document Store -> Kùzu).

Extraction only — not graph *queries* (chat/viewer read the graph elsewhere).
Diffs the Document Store against the `graph_state` ledger and converges the Kùzu
graph: run KG extraction on new/changed documents, delete graph data for
removed/changed ones. `embed_kg_nodes=False` keeps this pipeline off Qdrant.

    python -m rag.graph [--verbose] [--delay SECONDS]
"""

from __future__ import annotations

import argparse
import sys
import time

from llama_index.core import PropertyGraphIndex

from .. import cli, config, errors, runtime, stores
from ..documents import filter_changes, plan_changes, to_document
from ..htmllog import GraphLog, source_label

LEDGER = "graph_state"


def _triplet_key(t):
    """Identity of a (source, relation, target) triplet for set-diffing."""
    s, r, o = t
    return (getattr(s, "id", None), r.label, getattr(o, "id", None))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.graph", description="Extract a property graph into Kùzu."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                        help="pause between documents (LLM extraction is per-chunk).")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="process at most N new/changed documents this run.")
    parser.add_argument("--docs", default=None, metavar="PATTERN",
                        help="glob (*,?) on doc_id; only matching documents are processed.")
    return parser


def run_graph(cfg, verbose: bool = False, delay: float = 0.0,
              max_items: int | None = None, docs_pattern: str | None = None) -> None:
    store = stores.open_document_store(cfg)
    graph_store, db = stores.open_graph_store(cfg)
    log = GraphLog(stores.logs_dir(cfg) / "graph.html")
    try:
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            vector_store=None,
            llm=runtime.Settings.llm,
            embed_model=runtime.Settings.embed_model,
            kg_extractors=stores.build_extractors(cfg),
            embed_kg_nodes=False,
            use_async=False,
        )

        want = store.want_hashes()
        have = store.state(LEDGER)
        to_process, to_delete = plan_changes(want, have)
        to_process, to_delete = filter_changes(to_process, to_delete, docs_pattern, max_items)
        print(f"graph: {len(to_process)} to extract, {len(to_delete)} to delete "
              f"({len(want)} documents)")

        for doc_id in to_delete:
            index.delete_ref_doc(doc_id)
            store.unmark(LEDGER, doc_id)
            if verbose:
                print(f"  delete {doc_id}")

        for i, doc_id in enumerate(to_process, 1):
            if doc_id in have:  # changed -> clear old graph data first
                index.delete_ref_doc(doc_id)
            sdoc = store.get(doc_id)
            if verbose:
                print(f"  extract {i}/{len(to_process)}: {doc_id}")
            before = {_triplet_key(t) for t in graph_store.get_triplets()}
            with errors.endpoint_errors(f"({cfg.models.api_base})"):
                index.insert(to_document(sdoc))
            store.mark(LEDGER, doc_id, sdoc.doc_hash)
            # Back-query: diff the graph to find relations this document created.
            new_rels = [t for t in graph_store.get_triplets()
                        if _triplet_key(t) not in before]
            log.add(doc_id, source_label(sdoc), sdoc.text, relations=new_rels)
            if delay and i < len(to_process):
                time.sleep(delay)
        print("done.")
    finally:
        store.close()
        stores.close_graph_db(db)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load(".")
    runtime.configure(cfg)
    run_graph(cfg, verbose=args.verbose, delay=max(0.0, args.delay),
              max_items=args.max, docs_pattern=args.docs)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
