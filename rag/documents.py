"""Document Store — the canonical corpus (SQLite).

Single source of truth for the extracted documents. The embedding and graph
pipelines are pure functions of this store: they diff their own ledger against it
(`want` vs `have`, keyed by `doc_hash`) to decide what to add / update / delete.

Tables:
  documents(doc_id PK, extractor, local_id, text, metadata json, doc_hash, extracted_at)
  embed_state(doc_id PK, doc_hash)     -- ledger for the embedding pipeline
  graph_state(doc_id PK, doc_hash)     -- ledger for the graph pipeline

`doc_id` is global: "{extractor}:{local_id}". `local_id` need only be unique
within its extractor. `doc_hash` = hash(text + content metadata) — the sole
signal every downstream pipeline uses to detect change.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Document

from .errors import UserError

# metadata keys that are bookkeeping, not content (kept out of embeddings/LLM).
# local_id is not stored — it is doc_id with the "extractor:" prefix stripped.
AUTO_KEYS = ("doc_id", "extractor", "doc_hash")

# Substituted for blank document text wherever a doc is embedded/indexed, so no
# empty text reaches the embedders (empty vectors break e.g. GraphRAG/lancedb).
EMPTY_PLACEHOLDER = "[empty file]"


def content_or_placeholder(text: str) -> str:
    return text if (text or "").strip() else EMPTY_PLACEHOLDER

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    doc_id TEXT PRIMARY KEY,
    extractor TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    doc_hash TEXT NOT NULL,
    extracted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_extractor ON documents(extractor);
CREATE TABLE IF NOT EXISTS embed_state(doc_id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_state(doc_id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graphrag_state(doc_id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS locks(component TEXT, key TEXT, value TEXT, PRIMARY KEY(component, key));
"""

LEDGERS = ("embed_state", "graph_state", "graphrag_state")


@dataclass
class StoredDoc:
    doc_id: str
    extractor: str
    text: str
    metadata: dict
    doc_hash: str
    extracted_at: str


def compute_hash(text: str, metadata: dict) -> str:
    """Content hash: text + content metadata (bookkeeping keys excluded)."""
    content_meta = {k: v for k, v in sorted(metadata.items()) if k not in AUTO_KEYS}
    payload = json.dumps([text, content_meta], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_document(doc: StoredDoc) -> Document:
    """Build a LlamaIndex Document with the global doc_id as its stable id."""
    d = Document(text=content_or_placeholder(doc.text),
                 metadata=dict(doc.metadata), id_=doc.doc_id)
    for key in AUTO_KEYS:
        if key in d.metadata:
            d.excluded_embed_metadata_keys.append(key)
            d.excluded_llm_metadata_keys.append(key)
    return d


class DocumentStore:
    def __init__(self, path: str | Path, check_same_thread: bool = True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- documents --------------------------------------------------------

    def upsert(self, extractor: str, local_id: str, text: str, metadata: dict) -> bool:
        """Insert or update a document. Returns True if new/changed, False if same."""
        doc_id = f"{extractor}:{local_id}"
        meta = dict(metadata)
        meta.update(doc_id=doc_id, extractor=extractor)
        doc_hash = compute_hash(text, meta)
        meta["doc_hash"] = doc_hash

        row = self.conn.execute(
            "SELECT doc_hash FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is not None and row["doc_hash"] == doc_hash:
            return False
        self.conn.execute(
            "INSERT OR REPLACE INTO documents"
            "(doc_id, extractor, text, metadata, doc_hash, extracted_at)"
            " VALUES (?,?,?,?,?,?)",
            (doc_id, extractor, text, json.dumps(meta, default=str),
             doc_hash, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        return True

    def reconcile(self, extractor: str, seen_local_ids: set[str]) -> list[str]:
        """Delete docs of `extractor` whose local_id is no longer present.

        Returns the deleted doc_ids. Deletion is scoped to one extractor.
        """
        rows = self.conn.execute(
            "SELECT doc_id FROM documents WHERE extractor=?", (extractor,)
        ).fetchall()
        prefix = len(extractor) + 1  # strip "extractor:" to recover local_id
        deleted = [r["doc_id"] for r in rows if r["doc_id"][prefix:] not in seen_local_ids]
        for doc_id in deleted:
            self.delete(doc_id)
        return deleted

    def delete(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        self.conn.commit()

    def get(self, doc_id: str) -> StoredDoc | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return self._row(row) if row else None

    def all(self) -> list[StoredDoc]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [self._row(r) for r in rows]

    def want_hashes(self) -> dict[str, str]:
        return {r["doc_id"]: r["doc_hash"]
                for r in self.conn.execute("SELECT doc_id, doc_hash FROM documents")}

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    @staticmethod
    def _row(row: sqlite3.Row) -> StoredDoc:
        return StoredDoc(
            doc_id=row["doc_id"], extractor=row["extractor"],
            text=row["text"], metadata=json.loads(row["metadata"]),
            doc_hash=row["doc_hash"], extracted_at=row["extracted_at"],
        )

    # --- pipeline ledgers -------------------------------------------------

    def _check_ledger(self, table: str) -> None:
        if table not in LEDGERS:
            raise ValueError(f"unknown ledger {table!r}")

    def state(self, table: str) -> dict[str, str]:
        self._check_ledger(table)
        return {r["doc_id"]: r["doc_hash"]
                for r in self.conn.execute(f"SELECT doc_id, doc_hash FROM {table}")}

    def mark(self, table: str, doc_id: str, doc_hash: str) -> None:
        self._check_ledger(table)
        self.conn.execute(
            f"INSERT OR REPLACE INTO {table}(doc_id, doc_hash) VALUES (?,?)",
            (doc_id, doc_hash),
        )
        self.conn.commit()

    def unmark(self, table: str, doc_id: str) -> None:
        self._check_ledger(table)
        self.conn.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))
        self.conn.commit()

    # --- locked config ----------------------------------------------------

    def check_locked(self, component: str, values: dict) -> None:
        """Record a component's build-time config on first use; on later runs,
        raise if any locked value changed (would corrupt the built store).

        Locked = fixed once the store is built (e.g. embedding model / dim,
        chunking). Changeable config (chat model, endpoint) is not passed here.
        """
        stored = {r["key"]: r["value"]
                  for r in self.conn.execute(
                      "SELECT key, value FROM locks WHERE component=?", (component,))}
        mismatches = {k: (stored[k], str(v)) for k, v in values.items()
                      if k in stored and stored[k] != str(v)}
        if mismatches:
            detail = "; ".join(f"{k}: '{o}' → '{n}'" for k, (o, n) in mismatches.items())
            raise UserError(
                f"{component}: locked config changed ({detail}). These are fixed once "
                f"the store is built. To change them, wipe the store "
                f"(make clean-storage) and re-index."
            )
        # Record first-seen values (and any newly added keys).
        for k, v in values.items():
            if k not in stored:
                self.conn.execute(
                    "INSERT OR REPLACE INTO locks(component, key, value) VALUES (?,?,?)",
                    (component, k, str(v)),
                )
        self.conn.commit()


def plan_changes(want: dict[str, str], have: dict[str, str]):
    """Return (to_process, to_delete) doc_id sets from want vs have hashes."""
    to_process = [d for d, h in want.items() if have.get(d) != h]
    to_delete = [d for d in have if d not in want]
    return to_process, to_delete


def filter_changes(to_process, to_delete, docs_pattern: str | None = None,
                   max_items: int | None = None):
    """Scope a change set: glob (*,?) on doc_id, then cap new items to max."""
    import fnmatch
    if docs_pattern:
        to_process = [d for d in to_process if fnmatch.fnmatch(d, docs_pattern)]
        to_delete = [d for d in to_delete if fnmatch.fnmatch(d, docs_pattern)]
    if max_items is not None:
        to_process = to_process[:max_items]
    return to_process, to_delete
