"""Trace what chat sends to / gets back from the stores (not model thinking).

`enable(retriever)` wraps each sub-retriever of a PropertyGraphIndex retriever so
every retrieval prints the query it ran and the nodes it got back, abbreviated.
This surfaces the vector query (VectorContextRetriever -> Qdrant) and the graph
query (LLMSynonymRetriever -> property graph) separately. The LLM's internal
condensing/answering is NOT traced — only store round-trips.
"""

from __future__ import annotations

import sys

# Friendlier labels than the raw class names.
_LABELS = {
    "VectorContextRetriever": "vector",
    "LLMSynonymRetriever": "graph/synonym",
    "TextToCypherRetriever": "graph/cypher",
    "CypherTemplateRetriever": "graph/cypher",
}


def _abbrev(text: str, width: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _label(retriever) -> str:
    name = type(retriever).__name__
    return _LABELS.get(name, name)


def _trace_line(*parts: str) -> None:
    print("  ".join(parts), file=sys.stderr)


def _wrap(retriever) -> None:
    original = retriever.retrieve  # bound method
    label = _label(retriever)

    def traced(query, **kwargs):
        nodes = original(query, **kwargs)
        q = getattr(query, "query_str", query)
        _trace_line(f"[db:{label}]", f"query={_abbrev(q)!r}", f"-> {len(nodes)} nodes")
        for nws in nodes:
            score = f"{nws.score:.2f}" if nws.score is not None else "  -  "
            node_id = (getattr(nws.node, "node_id", "") or "")[:8]
            _trace_line(f"    · {score}", node_id, _abbrev(nws.node.get_content()))
        return nodes

    retriever.retrieve = traced  # BaseRetriever is not pydantic; instance attr is fine


def enable(retriever) -> int:
    """Wrap a retriever (and its sub-retrievers) to trace store queries.

    Returns how many retrievers were wrapped.
    """
    subs = getattr(retriever, "sub_retrievers", None)
    targets = list(subs) if subs else [retriever]
    for r in targets:
        _wrap(r)
    return len(targets)
