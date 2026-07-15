"""GraphRAG extraction pipeline (Document Store -> Microsoft GraphRAG index).

Runs Microsoft's GraphRAG over the extracted documents, storing its own index
(entities, relationships, communities, community reports) as parquet under
storage/graphrag/output — a separate DB, NOT unified with Kùzu. Same CLI flags
as rag.graph.

    python -m rag.graphrag [--verbose] [--delay SECONDS] [--max N] [--docs PATTERN]

Note: GraphRAG re-indexes its corpus each run (no per-doc surgical delete). The
--docs / --max flags select which documents are fed in; --delay is accepted for
parity but GraphRAG manages its own request pacing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .. import cli, config, runtime, stores
from ..documents import content_or_placeholder, filter_changes, plan_changes
from ..errors import UserError
from ..htmllog import source_label
from .workspace import ensure_workspace, output_dir

LEDGER = "graphrag_state"
# GraphRAG's community-report step crashes on a sparse graph (few/no relationships,
# as with prose notes): empty communities -> KeyError 'community'. We run a reduced
# pipeline that builds documents, text units, entities and relationships but skips
# the community layer (and claims/embeddings we don't use), so indexing completes.
# The graph (entities + relationships + provenance) is what the viewer reads.
_FULL_WORKFLOWS = [
    "load_input_documents",
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "create_final_text_units",
    "generate_text_embeddings",   # entity + text-unit embeddings (its own vector search)
]
_UPDATE_WORKFLOWS = [
    "load_update_documents",
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "create_final_text_units",
    "generate_text_embeddings",
    "update_final_documents",
    "update_entities_relationships",
    "update_text_units",
    "update_text_embeddings",
    "update_clean_state",
]
# The optional community layer, computed on demand over the existing graph, plus
# embedding the community reports so GraphRAG's search includes them.
_COMMUNITY_WORKFLOWS = [
    "create_communities", "create_community_reports", "generate_text_embeddings",
]
# What each run embeds (community reports only exist after --communities).
_GRAPH_EMBEDDINGS = ["entity_description", "text_unit_text"]
_COMMUNITY_EMBEDDINGS = ["community_full_content"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.graphrag", description="Index documents with Microsoft GraphRAG."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                        help="accepted for parity; GraphRAG paces its own requests.")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="index at most N new/changed documents this run.")
    parser.add_argument("--docs", default=None, metavar="PATTERN",
                        help="glob (*,?) on doc_id; only matching documents indexed.")
    parser.add_argument("--rebuild", action="store_true",
                        help="full re-index of all documents (also applies deletions).")
    parser.add_argument("--communities", action="store_true",
                        help="(re)compute the community layer over the existing graph; "
                             "run periodically, not every import.")
    return parser


def run_communities(cfg, verbose: bool = False) -> None:
    """Compute communities + community reports over the existing entities/relationships.

    Cheap re-extraction-free: clustering is free, reports cost one LLM call per
    community. Fails if the graph is too sparse to form communities.
    """
    from graphrag.api import build_index
    from graphrag.config.load_config import load_config

    if not _has_previous(cfg):
        raise UserError("No graph index yet — run `rag.graphrag` to build "
                        "entities/relationships before computing communities.")
    root = ensure_workspace(cfg)
    gr_config = load_config(root)
    gr_config.workflows = _COMMUNITY_WORKFLOWS
    gr_config.embed_text.names = _COMMUNITY_EMBEDDINGS
    _sanitize_graphrag_embeddings()
    print("graphrag: (re)computing communities over the current graph")
    results = asyncio.run(build_index(
        gr_config, input_documents=None, is_update_run=False, verbose=verbose))
    errors = [r for r in results if getattr(r, "error", None)]
    for r in errors:
        wf = getattr(r, "workflow", "?")
        err = r.error
        kind = type(err).__name__ if isinstance(err, BaseException) else "error"
        print(f"  ERROR in GraphRAG workflow '{wf}': {kind}: {err}", file=sys.stderr)
    if errors:
        print("community build FAILED (likely too few relationships to form "
              "communities yet). The entities/relationships graph is unaffected.",
              file=sys.stderr)
        return
    print("done.")


def _frame(store, doc_ids):
    """GraphRAG documents DataFrame (id, text, title) for the given doc_ids."""
    import pandas as pd

    rows = []
    for doc_id in doc_ids:
        sdoc = store.get(doc_id)
        if sdoc is not None:
            rows.append({"id": sdoc.doc_id, "text": content_or_placeholder(sdoc.text),
                         "title": source_label(sdoc)})
    return pd.DataFrame(rows, columns=["id", "text", "title"])


def _sanitize_graphrag_embeddings() -> None:
    """Replace empty/whitespace embedding inputs with a placeholder (extends the
    empty-document rule to GraphRAG's internal embed step).

    GraphRAG embeds entity/text-unit/community text straight through litellm; an
    empty string (e.g. an entity with no description) makes our endpoint return
    400, yielding a bad vector and a lancedb 'list_size' crash. Any non-empty
    string embeds fine, so map empties to '[empty]'.
    """
    import graphrag.index.operations.embed_text.embed_text as et
    if getattr(et, "_rag_sanitized", False):
        return
    original = et.run_embed_text

    async def wrapped(texts, *args, **kwargs):
        texts = [t if (t or "").strip() else "[empty]" for t in texts]
        return await original(texts, *args, **kwargs)

    et.run_embed_text = wrapped
    et._rag_sanitized = True


def _has_previous(cfg) -> bool:
    # entities.parquet is the last table our reduced pipeline exports; its presence
    # marks a complete prior index (outputs are exported only on success, so a
    # crashed run leaves none -> we correctly retry as a full build).
    return (output_dir(cfg) / "entities.parquet").exists()


def run_graphrag(cfg, verbose: bool = False, max_items: int | None = None,
                 docs_pattern: str | None = None, rebuild: bool = False) -> None:
    from graphrag.api import build_index
    from graphrag.config.load_config import load_config

    store = stores.open_document_store(cfg)
    store.check_locked("graphrag", {
        "embed_model": cfg.models.embed_model,
        "chunk_size": cfg.chunking.chunk_size,
        "chunk_overlap": cfg.chunking.chunk_overlap,
    })
    try:
        want = store.want_hashes()
        have = store.state(LEDGER)

        if rebuild or not _has_previous(cfg):
            # Full (re)build: index everything, from scratch. is_update_run=False.
            to_process = list(want)
            to_delete = []
            is_update = False
        else:
            to_process, to_delete = plan_changes(want, have)
            is_update = True
        to_process, _ = filter_changes(to_process, to_delete, docs_pattern, max_items)

        print(f"graphrag: {len(to_process)} to index "
              f"({'update' if is_update else 'full build'}), "
              f"{len(to_delete)} deleted, {len(want)} documents")

        if to_delete and is_update:
            print("  note: GraphRAG update does not remove documents from the index; "
                  "run --rebuild to drop deleted documents.", file=sys.stderr)

        if to_process:
            if verbose:
                for i, doc_id in enumerate(to_process, 1):
                    print(f"  indexing {i}/{len(to_process)}: {doc_id}")
            frame = _frame(store, to_process)
            root = ensure_workspace(cfg)
            gr_config = load_config(root)
            # Reduced pipeline (skip the community layer that crashes on sparse graphs).
            gr_config.workflows = _UPDATE_WORKFLOWS if is_update else _FULL_WORKFLOWS
            gr_config.embed_text.names = _GRAPH_EMBEDDINGS
            _sanitize_graphrag_embeddings()
            results = asyncio.run(build_index(
                gr_config, input_documents=frame, is_update_run=is_update, verbose=verbose))
            errors = [r for r in results if getattr(r, "error", None)]
            for r in errors:
                wf = getattr(r, "workflow", "?")
                err = r.error
                kind = type(err).__name__ if isinstance(err, BaseException) else "error"
                print(f"  ERROR in GraphRAG workflow '{wf}': {kind}: {err}",
                      file=sys.stderr)
            if errors:
                print(f"GraphRAG indexing FAILED ({len(errors)} workflow error(s)). "
                      f"Ledger not updated — safe to re-run.", file=sys.stderr)
                return
        elif not to_delete:
            print("nothing to index.")

        for doc_id in to_process:
            store.mark(LEDGER, doc_id, want[doc_id])
        for doc_id in to_delete:
            store.unmark(LEDGER, doc_id)
        if rebuild:  # full build supersedes any stale ledger rows for gone docs
            for doc_id in list(have):
                if doc_id not in want:
                    store.unmark(LEDGER, doc_id)
        print("done.")
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load(".")
    runtime.configure(cfg)
    runtime.verify_embedding_dim(cfg)
    if args.communities:
        run_communities(cfg, verbose=args.verbose)
    else:
        run_graphrag(cfg, verbose=args.verbose, max_items=args.max,
                     docs_pattern=args.docs, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
