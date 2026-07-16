"""SQLite metadata database: connection, WAL, schema and migrations (PLAN §3).

The database holds only metadata and blob references; blobs live on the
filesystem (:mod:`rag.blobs`). It runs in WAL mode so chat (readers) can run
while the pipeline (a writer) works. The schema is versioned via SQLite's
built-in ``PRAGMA user_version`` and evolved by an ordered, forward-only list of
migrations applied automatically on open.

Metadata is stored as JSONB (SQLite >= 3.45). Commonly-filtered keys (date,
title, mime-type) are exposed as generated, indexed columns on ``documents``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

# Milliseconds a blocked writer waits for the lock before raising SQLITE_BUSY.
# Makes the occasional concurrent writer queue instead of erroring.
_BUSY_TIMEOUT_MS: int = 5000

# Migration 0001: create the entire initial schema. Kept as one statement block
# so a fresh database is never in an un-versioned/partial state.
_MIGRATION_0001_INIT: str = """
-- Extraction output: one row per live source document (PLAN §3.2).
CREATE TABLE documents (
    source_type TEXT    NOT NULL,               -- plugin id (== plugin subdir name)
    doc_id      TEXT    NOT NULL,               -- unique within a source_type
    version     INTEGER NOT NULL,               -- bumps when content OR stable metadata changes
    blob_hash   TEXT    NOT NULL,               -- reference to the raw source blob
    -- Content-defining metadata: a change bumps the version and retriggers the
    -- downstream pipeline. JSONB; source of truth.
    stable_metadata    BLOB NOT NULL,
    -- Informational, mutable state (IMAP folder, read flag, …) that must NOT
    -- affect the version. Written on every upsert; never retriggers pipelines.
    transient_metadata BLOB NOT NULL,
    updated_at  TEXT    NOT NULL,               -- ISO timestamp of last extraction touch
    -- Generated, indexed columns for the common filter keys, taken from the
    -- STABLE metadata; always in sync by construction (computed on read).
    date      TEXT GENERATED ALWAYS AS (stable_metadata ->> 'date')      VIRTUAL,
    title     TEXT GENERATED ALWAYS AS (stable_metadata ->> 'title')     VIRTUAL,
    mime_type TEXT GENERATED ALWAYS AS (stable_metadata ->> 'mime-type') VIRTUAL,
    PRIMARY KEY (source_type, doc_id)
);
CREATE INDEX idx_documents_date ON documents(date);
CREATE INDEX idx_documents_mime ON documents(mime_type);

-- Processing output: one row per translation run of a document (PLAN §3.2).
CREATE TABLE translation_sets (
    id             INTEGER PRIMARY KEY,
    source_type    TEXT    NOT NULL,
    doc_id         TEXT    NOT NULL,
    source_version INTEGER NOT NULL,            -- documents.version this run was made from
    metadata       BLOB    NOT NULL,            -- JSONB run-level pipeline info
    translated_at  TEXT    NOT NULL,
    git_rev        TEXT    NOT NULL,
    is_active      INTEGER NOT NULL DEFAULT 0,  -- applies to the WHOLE set
    FOREIGN KEY (source_type, doc_id)
        REFERENCES documents(source_type, doc_id) ON DELETE CASCADE
);
-- At most one active set per document; zero active is a valid state.
CREATE UNIQUE INDEX idx_active_set
    ON translation_sets(source_type, doc_id) WHERE is_active = 1;

"""

# The translation_parts and embeddings tables — kept as standalone constants so
# both the initial migration and the catch-up migration 0002 create the SAME
# schema (single source of truth, no drift).

# One row per output blob ("part") within a translation set (PLAN §3.2, §5.1).
_DDL_TRANSLATION_PARTS: str = """
CREATE TABLE translation_parts (
    id         INTEGER PRIMARY KEY,
    set_id     INTEGER NOT NULL REFERENCES translation_sets(id) ON DELETE CASCADE,
    -- The part's sub_doc_id: a JSONB list of strings identifying the part within
    -- its document (e.g. ["body"], ["attachment:1","body"]). Source of truth.
    sub_doc_id BLOB    NOT NULL,
    blob_hash  TEXT    NOT NULL,                -- reference to this part's translated blob
    metadata   BLOB    NOT NULL,               -- JSONB per-part metadata
    -- Canonical text encoding of sub_doc_id, used for the uniqueness constraint
    -- (and lookups). Generated from the JSONB so it always matches by construction.
    part_key   TEXT GENERATED ALWAYS AS (json(sub_doc_id)) VIRTUAL,
    UNIQUE (set_id, part_key)
);
"""

# Join of parts x embedding models: one row per (part, model) embedded, so
# staleness is tracked per model (PLAN §3.2, §6).
_DDL_EMBEDDINGS: str = """
CREATE TABLE embeddings (
    source_type TEXT    NOT NULL,
    doc_id      TEXT    NOT NULL,
    set_id      INTEGER NOT NULL REFERENCES translation_sets(id) ON DELETE CASCADE,
    part_id     INTEGER NOT NULL REFERENCES translation_parts(id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,               -- embedding model id (own index/collection)
    embedded_at TEXT    NOT NULL,
    -- A part is embedded at most once per model.
    PRIMARY KEY (part_id, model)
);
"""


def _migration_0001(conn: sqlite3.Connection) -> None:
    """Apply migration 0001 — create the initial schema."""
    # documents + translation_sets, then the shared parts/embeddings DDL.
    conn.executescript(_MIGRATION_0001_INIT)
    conn.executescript(_DDL_TRANSLATION_PARTS)
    conn.executescript(_DDL_EMBEDDINGS)


def _migration_0002(conn: sqlite3.Connection) -> None:
    """Recreate translation_parts + embeddings with the current schema.

    Catch-up for databases created before ``translation_parts`` stored
    ``sub_doc_id`` (and ``embeddings`` used the ``model`` column). Both tables are
    empty in practice (no translations/embeddings existed under the old schema),
    so dropping and recreating them is safe and loses nothing. On a fresh database
    this simply rebuilds the identical tables.
    """
    # Drop in FK order: embeddings references translation_parts.
    conn.executescript("DROP TABLE IF EXISTS embeddings; DROP TABLE IF EXISTS translation_parts;")
    conn.executescript(_DDL_TRANSLATION_PARTS)
    conn.executescript(_DDL_EMBEDDINGS)


def _migration_0003(conn: sqlite3.Connection) -> None:
    """Create ``embedding_models`` — a per-model config fingerprint guard (PLAN §6).

    Records the incompatibility-defining config of each embedding model (its
    identity/type/dim/metric). The embed pass refuses to run if a model's current
    config differs from what its collection was built with.
    """
    conn.executescript(
        "CREATE TABLE embedding_models ("
        "  model       TEXT NOT NULL PRIMARY KEY,"  # the model id
        "  fingerprint TEXT NOT NULL,"              # canonical incompatibility fingerprint
        "  created_at  TEXT NOT NULL"
        ");"
    )


# Ordered list of migrations. Index i (0-based) upgrades user_version from i to
# i+1. Appending a new migration is the only supported way to change the schema.
_MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migration_0001,
    _migration_0002,
    _migration_0003,
]


class Database:
    """Owns a SQLite connection configured for the pipeline."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Wrap an already-configured connection (use :meth:`open` normally)."""
        # The live SQLite connection; callers may use it for ad-hoc queries.
        self.conn: sqlite3.Connection = conn

    @classmethod
    def open(cls, path: Path) -> "Database":
        """Open (creating if needed) the database at ``path`` and migrate it.

        Configures WAL, a busy timeout and foreign-key enforcement, then applies
        any pending migrations. Refuses to open a database whose ``user_version``
        is newer than this build understands.
        """
        # Ensure the parent directory exists before SQLite creates the file.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Rows come back as sqlite3.Row for name-based access throughout.
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cls._configure(conn)
        cls._migrate(conn)
        return cls(conn)

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        """Apply the per-connection pragmas the pipeline relies on."""
        # WAL: many concurrent readers + one writer across processes.
        conn.execute("PRAGMA journal_mode=WAL")
        # Wait rather than error when another writer holds the lock briefly.
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        # Enforce ON DELETE CASCADE for the document -> translations -> embeddings
        # chain (off by default in SQLite).
        conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Apply pending migrations, each in its own transaction.

        Reads the current ``user_version`` and applies every migration beyond it
        in order, bumping the version inside the same transaction so a failure
        rolls back cleanly and never leaves a half-migrated database.
        """
        # Current on-disk schema version (0 for a brand-new database).
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        target = len(_MIGRATIONS)
        if current > target:
            # DB was upgraded by a newer build; refuse rather than misread it.
            raise RuntimeError(
                f"database schema version {current} is newer than supported {target}; "
                "upgrade the software or restore a compatible backup"
            )
        # Apply each outstanding migration in ascending order.
        for version in range(current, target):
            migration = _MIGRATIONS[version]
            with conn:  # BEGIN/COMMIT — atomic per migration step.
                migration(conn)
                # PRAGMA cannot be parameterised, so interpolate the trusted int.
                conn.execute(f"PRAGMA user_version={version + 1}")

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()

    def __enter__(self) -> "Database":
        """Enter a ``with`` block, returning this database."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection on exit from a ``with`` block."""
        self.close()
