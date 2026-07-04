"""Extractor scaffolding: shared DB handling + reconciliation.

An extractor only scans its source and yields `(local_id, text, metadata)` per
object. The scaffolding does everything else: prefix the extractor name, upsert
into the Document Store, and after the scan delete any of *that extractor's*
documents whose `local_id` was not seen this run (deletion, scoped per extractor).
"""

from __future__ import annotations

from typing import Iterable, Protocol

from ..documents import DocumentStore


class Extractor(Protocol):
    name: str

    def enabled(self, cfg) -> bool: ...

    def iter_documents(self, cfg) -> Iterable[tuple[str, str, dict]]:
        """Yield (local_id, text, metadata) for each source object."""


def run_extractor(extractor: Extractor, cfg, store: DocumentStore,
                  verbose: bool = False) -> dict:
    """Scan one extractor into the Document Store; reconcile deletions."""
    seen: set[str] = set()
    new = 0
    total = 0
    for local_id, text, metadata in extractor.iter_documents(cfg):
        seen.add(local_id)
        total += 1
        if store.upsert(extractor.name, local_id, text, metadata):
            new += 1
            if verbose:
                print(f"  [{extractor.name}] upsert {local_id}")
    deleted = store.reconcile(extractor.name, seen)
    if verbose:
        for doc_id in deleted:
            print(f"  [{extractor.name}] delete {doc_id}")
    print(f"{extractor.name}: {total} scanned, {new} new/changed, {len(deleted)} deleted")
    return {"total": total, "new": new, "deleted": len(deleted)}
