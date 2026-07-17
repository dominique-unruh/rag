"""Orchestration of an extraction run (PLAN §4).

Wires together the config, secrets, database, blob store and per-plugin state
directory, then invokes the selected plugin's :meth:`Extractor.run`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..blobs import BlobStore
from ..config import Config
from ..db import Database
from ..secrets import Secrets
from .discovery import load_plugin
from .store import SqliteStore

# Subdirectory (under the storage dir) holding each plugin's private state dir.
_PLUGIN_STATE_SUBDIR: str = "plugin_state"
# Filename of the SQLite metadata database within the storage dir.
_DB_FILENAME: str = "metadata.db"
# Subdirectory (under the storage dir) holding the content-addressed blobs.
_BLOBS_SUBDIR: str = "blobs"


def _state_dir_for(storage_dir: Path, source_type: str) -> Path:
    """Return (creating) the private state directory for ``source_type``.

    The directory is stable across runs and unique to the plugin; its contents
    are opaque to the framework (PLAN §4.1).
    """
    state_dir = storage_dir / _PLUGIN_STATE_SUBDIR / source_type
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def run_extraction(
    source_type: str,
    options: dict[str, Any],
    config: Config,
    *,
    secrets: Secrets | None = None,
    db: Database | None = None,
) -> None:
    """Run the extraction plugin ``source_type`` with ``options``.

    Opens the shared database and blob store under the configured storage
    directory (unless a ``db`` is supplied, e.g. by tests), builds a scoped
    :class:`SqliteStore`, and calls the plugin. Any database opened here is
    closed again before returning.
    """
    storage_dir = config.storage_dir
    # Resolve the plugin class from its subdirectory name (also validates it).
    plugin_cls = load_plugin(source_type)

    # Blob store and per-plugin state directory live under the storage dir.
    blobs = BlobStore(storage_dir / _BLOBS_SUBDIR)
    # One-time upgrade of any pre-marker blobs to the compressed format (no-op once
    # done). A pipeline run is a natural, single-writer moment to do this.
    blobs.migrate()
    state_dir = _state_dir_for(storage_dir, source_type)
    # Default the secret resolver if the caller did not inject one.
    resolver = secrets if secrets is not None else Secrets()

    # Open the DB unless one was injected; track ownership so we only close ours.
    owns_db = db is None
    database = db if db is not None else Database.open(storage_dir / _DB_FILENAME)
    try:
        # Store is bound to this source_type -> enforces plugin isolation.
        store = SqliteStore(database.conn, blobs, source_type)
        plugin = plugin_cls()
        # Hand the plugin only its own config subtree and the shared secrets.
        plugin.run(store, state_dir, config.plugin_config(source_type), resolver, **options)
    finally:
        # Close only a database we opened ourselves.
        if owns_db:
            database.close()
