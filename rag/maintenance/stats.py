"""stats: count everything in the metadata DB, plus per-plugin state stats (PLAN §8).

A read-only maintenance tool. It reports, from the SQLite metadata database:
documents (total + per source_type); translation sets (total/active), documents
with no active translation, active sets that have gone stale (their
``source_version`` no longer matches the document's current ``version``), and
translation parts (total + those in active sets); and, per embedding model, how
many active parts are embedded vs still missing plus the raw embedding-row count
(flagging models present in the DB but absent from the config).

It also asks each extraction plugin for its own cheap statistics via
:meth:`rag.extract.base.Extractor.statistics` — those read only the plugin's
already-stored state dir (no network, no IMAP, no DB scans) — and reports on-disk
storage use of the metadata DB (with WAL sidecars), the blob store (actual disk
allocation via ``du``), the vector store directory, and each plugin's state dir.

The tool never writes; it opens the database read-only.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from ..config import Config
from ..extract.discovery import discover_plugins

# Subdirectory (under the storage dir) holding each plugin's private state dir —
# mirrors :data:`rag.extract.runner._PLUGIN_STATE_SUBDIR`. Kept in sync here so
# stats can locate a plugin's state without *creating* it (runner's helper does).
_PLUGIN_STATE_SUBDIR: str = "plugin_state"

# Filename of the SQLite metadata database within the storage dir.
_DB_FILENAME: str = "metadata.db"

# Subdirectory (under the storage dir) holding the embedded-Qdrant vector store —
# mirrors the path used by embed/dbbrowser (``config.storage_dir / "qdrant"``).
_QDRANT_SUBDIR: str = "qdrant"

# Subdirectory (under the storage dir) holding the content-addressed blobs —
# mirrors :data:`rag.extract.runner._BLOBS_SUBDIR`.
_BLOBS_SUBDIR: str = "blobs"


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open the metadata database read-only for counting.

    Uses a ``mode=ro`` URI so the tool can never mutate the DB; rows come back as
    :class:`sqlite3.Row` for name access.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _dir_bytes(path: Path) -> int:
    """Return the total size in bytes of all files under ``path`` (0 if absent).

    Walks the tree and sums regular-file sizes; symlinks/unreadable entries are
    skipped so a stray link never inflates or crashes the count.
    """
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        # Follow only real files; skip symlinks and anything that errors on stat.
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _du_bytes(path: Path) -> int:
    """Return the *actual disk usage* of ``path`` (block-allocated) via ``du``.

    Unlike :func:`_dir_bytes` (which sums apparent file sizes), ``du`` reports the
    real block allocation — the right figure for the blob store, whose many small
    files are each rounded up to a filesystem block. One ``du -sk`` also beats
    walking a huge blob tree in Python. Falls back to :func:`_dir_bytes` when
    ``du`` is unavailable or errors (e.g. non-POSIX platforms).
    """
    if not path.exists():
        return 0
    du = shutil.which("du")
    if du is not None:
        try:
            # ``-s`` summarise, ``-k`` report in 1024-byte blocks (POSIX-portable).
            out = subprocess.run(
                [du, "-sk", str(path)], capture_output=True, text=True, check=True
            )
            # Output is "<kib>\t<path>"; the leading integer is KiB used.
            return int(out.stdout.split()[0]) * 1024
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            # Any failure degrades to the pure-Python apparent-size sum.
            pass
    return _dir_bytes(path)


def _db_bytes(db_path: Path) -> int:
    """Return the on-disk size of the SQLite DB including its WAL/SHM sidecars.

    WAL mode leaves ``-wal``/``-shm`` companion files that are part of the DB's
    real footprint, so they are summed alongside the main file.
    """
    total = 0
    # The main db file plus its two WAL sidecars (each present only sometimes).
    for name in (db_path.name, db_path.name + "-wal", db_path.name + "-shm"):
        sidecar = db_path.with_name(name)
        if sidecar.is_file():
            total += sidecar.stat().st_size
    return total


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    """Run a single-value counting query and return it as an int (0 if NULL/none)."""
    row = conn.execute(sql, params).fetchone()
    # COUNT(*) always yields a row; guard the value being NULL just in case.
    return int(row[0]) if row is not None and row[0] is not None else 0


def _source_clause(
    source_type: str | None, *, first: bool = False, column: str = "source_type"
) -> tuple[str, tuple[Any, ...]]:
    """Build an optional ``source_type`` filter fragment and its parameters.

    Returns ``("", ())`` when no source is requested; otherwise a ``WHERE``/``AND``
    fragment (``first`` picks the leading keyword) filtering ``column`` (qualify it
    like ``ts.source_type`` when the query joins tables), plus the parameter tuple.
    """
    if source_type is None:
        return "", ()
    keyword = "WHERE" if first else "AND"
    return f" {keyword} {column} = ?", (source_type,)


def _document_stats(conn: sqlite3.Connection, source_type: str | None) -> dict[str, Any]:
    """Count documents overall and per source_type (respecting the source filter)."""
    clause, params = _source_clause(source_type, first=True)
    total = _scalar(conn, f"SELECT COUNT(*) FROM documents{clause}", params)
    # Per-source breakdown; the same optional filter keeps it to one source.
    by_source: dict[str, int] = {}
    for row in conn.execute(
        f"SELECT source_type, COUNT(*) AS n FROM documents{clause} "
        "GROUP BY source_type ORDER BY source_type",
        params,
    ):
        by_source[row["source_type"]] = int(row["n"])
    return {"total": total, "by_source": by_source}


def _translation_stats(conn: sqlite3.Connection, source_type: str | None) -> dict[str, Any]:
    """Count translation sets/parts and the derived "missing/stale" figures."""
    # `ts.`-qualified filter for queries whose primary table is translation_sets,
    # and a `d.`-qualified filter for the untranslated-documents count.
    ts_clause, ts_params = _source_clause(source_type, column="ts.source_type")
    doc_clause, doc_params = _source_clause(source_type, column="d.source_type")

    sets_total = _scalar(
        conn,
        "SELECT COUNT(*) FROM translation_sets ts WHERE 1=1" + ts_clause,
        ts_params,
    )
    sets_active = _scalar(
        conn,
        "SELECT COUNT(*) FROM translation_sets ts WHERE ts.is_active = 1" + ts_clause,
        ts_params,
    )
    # Documents lacking ANY active translation set — the untranslated backlog.
    untranslated = _scalar(
        conn,
        "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS ("
        "  SELECT 1 FROM translation_sets ts"
        "  WHERE ts.source_type = d.source_type AND ts.doc_id = d.doc_id AND ts.is_active = 1"
        ")" + doc_clause,
        doc_params,
    )
    # Active sets whose source_version no longer matches the document's version —
    # a translation that needs redoing (the document changed under it).
    stale_active = _scalar(
        conn,
        "SELECT COUNT(*) FROM translation_sets ts"
        " JOIN documents d ON d.source_type = ts.source_type AND d.doc_id = ts.doc_id"
        " WHERE ts.is_active = 1 AND ts.source_version != d.version" + ts_clause,
        ts_params,
    )
    # Translation parts overall, and those belonging to an active set (the parts
    # the embed stage actually cares about).
    parts_total = _scalar(
        conn,
        "SELECT COUNT(*) FROM translation_parts tp"
        " JOIN translation_sets ts ON ts.id = tp.set_id WHERE 1=1" + ts_clause,
        ts_params,
    )
    parts_active = _scalar(
        conn,
        "SELECT COUNT(*) FROM translation_parts tp"
        " JOIN translation_sets ts ON ts.id = tp.set_id WHERE ts.is_active = 1" + ts_clause,
        ts_params,
    )
    return {
        "sets_total": sets_total,
        "sets_active": sets_active,
        "untranslated_documents": untranslated,
        "stale_active_sets": stale_active,
        "parts_total": parts_total,
        "parts_active": parts_active,
    }


def _embedding_stats(
    conn: sqlite3.Connection, config: Config, source_type: str | None, parts_active: int
) -> dict[str, Any]:
    """Per-model embedded/missing active-part counts + raw row totals.

    ``parts_active`` is the number of parts in active sets (from
    :func:`_translation_stats`); a model's *missing* active parts is that total
    minus the ones it has already embedded. Models appearing in the DB but not in
    the current config are reported with ``configured: False``.
    """
    from ..embed.models import load_models

    ts_clause, ts_params = _source_clause(source_type, column="ts.source_type")

    # Configured models (registry) plus any model id that shows up in embeddings.
    configured = set(load_models(config.raw))
    # Raw embedding-row totals per model (optionally scoped to one source_type).
    emb_clause, emb_params = _source_clause(source_type, first=True)
    rows_by_model: dict[str, int] = {}
    for row in conn.execute(
        f"SELECT model, COUNT(*) AS n FROM embeddings{emb_clause} GROUP BY model",
        emb_params,
    ):
        rows_by_model[row["model"]] = int(row["n"])

    models: dict[str, Any] = {}
    # Union: report every configured model (even at zero) and every stray model.
    for model_id in sorted(configured | set(rows_by_model)):
        # Active parts already embedded for this model.
        embedded_active = _scalar(
            conn,
            "SELECT COUNT(*) FROM translation_parts tp"
            " JOIN translation_sets ts ON ts.id = tp.set_id AND ts.is_active = 1"
            " JOIN embeddings e ON e.part_id = tp.id AND e.model = ?"
            " WHERE 1=1" + ts_clause,
            (model_id, *ts_params),
        )
        models[model_id] = {
            "configured": model_id in configured,
            "rows_total": rows_by_model.get(model_id, 0),
            "embedded_active_parts": embedded_active,
            # What still needs embedding for this model to cover the active set.
            "missing_active_parts": max(0, parts_active - embedded_active),
        }
    return {"models": models}


def _plugin_stats(config: Config, source_type: str | None) -> dict[str, Any]:
    """Collect each extraction plugin's own cheap ``statistics()`` output.

    Reads only already-persisted plugin state (the plugin contract). A plugin with
    no state dir yet is skipped; a plugin that raises is reported with its error
    rather than aborting the whole stats run.
    """
    storage_dir = config.storage_dir
    out: dict[str, Any] = {}
    for st, plugin_cls in sorted(discover_plugins().items()):
        # Honour a single-source filter when given.
        if source_type is not None and st != source_type:
            continue
        # Locate (do NOT create) the plugin's private state dir; skip if unused.
        state_dir = storage_dir / _PLUGIN_STATE_SUBDIR / st
        if not state_dir.is_dir():
            continue
        try:
            out[st] = plugin_cls().statistics(state_dir, config.plugin_config(st))
        except Exception as exc:  # a plugin's stats must never crash the tool
            out[st] = {"error": str(exc)}
    return out


def _storage_stats(config: Config, source_type: str | None) -> dict[str, Any]:
    """Report on-disk storage use of the DB, vector store and each plugin dir.

    Sizes are in bytes: the metadata DB (with WAL sidecars), the embedded-Qdrant
    vector store directory, and each extraction plugin's state dir. With a
    ``source_type`` filter, only that plugin's dir is reported (the DB, blobs and
    vector store are shared, so they are always shown).
    """
    storage_dir = config.storage_dir
    plugins: dict[str, int] = {}
    for st in sorted(discover_plugins()):
        # Honour a single-source filter when given.
        if source_type is not None and st != source_type:
            continue
        plugin_dir = storage_dir / _PLUGIN_STATE_SUBDIR / st
        # Only report plugins that actually have a state dir on disk.
        if plugin_dir.is_dir():
            plugins[st] = _dir_bytes(plugin_dir)
    return {
        "database_bytes": _db_bytes(storage_dir / _DB_FILENAME),
        # Blobs: real disk allocation (du), not apparent size — many small files.
        "blobs_bytes": _du_bytes(storage_dir / _BLOBS_SUBDIR),
        "vector_store_bytes": _dir_bytes(storage_dir / _QDRANT_SUBDIR),
        "plugins": plugins,
    }


def gather_stats(config: Config, *, source_type: str | None = None) -> dict[str, Any]:
    """Gather all statistics into one nested mapping.

    Opens the metadata DB read-only, counts documents/translations/embeddings
    (optionally restricted to one ``source_type``), then folds in each plugin's
    :meth:`~rag.extract.base.Extractor.statistics`. Never writes anything.
    """
    db_path = config.storage_dir / _DB_FILENAME
    documents: dict[str, Any]
    translations: dict[str, Any]
    embeddings: dict[str, Any]
    if db_path.exists():
        conn = _open_ro(db_path)
        try:
            documents = _document_stats(conn, source_type)
            translations = _translation_stats(conn, source_type)
            embeddings = _embedding_stats(conn, config, source_type, translations["parts_active"])
        finally:
            conn.close()
    else:
        # No database yet (nothing extracted): report empty rather than erroring.
        documents = {"total": 0, "by_source": {}}
        translations = {
            "sets_total": 0, "sets_active": 0, "untranslated_documents": 0,
            "stale_active_sets": 0, "parts_total": 0, "parts_active": 0,
        }
        embeddings = {"models": {}}
    return {
        "documents": documents,
        "translations": translations,
        "embeddings": embeddings,
        "plugins": _plugin_stats(config, source_type),
        "storage": _storage_stats(config, source_type),
    }


def _render(stats: dict[str, Any]) -> str:
    """Render the gathered stats as a human-readable, indented text report."""
    lines: list[str] = []

    # --- documents ---
    docs = stats["documents"]
    lines.append(f"Documents: {docs['total']}")
    for st, n in docs["by_source"].items():
        lines.append(f"  {st}: {n}")

    # --- translations ---
    tr = stats["translations"]
    lines.append("Translations:")
    lines.append(f"  sets: {tr['sets_total']} ({tr['sets_active']} active)")
    lines.append(f"  parts: {tr['parts_total']} ({tr['parts_active']} in active sets)")
    lines.append(f"  documents without active translation: {tr['untranslated_documents']}")
    lines.append(f"  stale active sets (source changed): {tr['stale_active_sets']}")

    # --- embeddings ---
    models = stats["embeddings"]["models"]
    lines.append("Embeddings:")
    if not models:
        lines.append("  (no models)")
    for model_id, m in models.items():
        # Flag models found in the DB but no longer in the config.
        tag = "" if m["configured"] else " [not in config]"
        lines.append(f"  {model_id}{tag}:")
        lines.append(f"    rows: {m['rows_total']}")
        lines.append(
            f"    active parts embedded: {m['embedded_active_parts']} "
            f"(missing: {m['missing_active_parts']})"
        )

    # --- per-plugin state stats ---
    plugins = stats["plugins"]
    lines.append("Plugins:")
    if not plugins:
        lines.append("  (none with stored state)")
    for st, pstats in plugins.items():
        lines.append(f"  {st}:")
        # Render the (possibly nested) plugin mapping with a small indenter.
        lines.extend(_render_mapping(pstats, indent=4))

    # --- storage use ---
    storage = stats["storage"]
    lines.append("Storage:")
    lines.append(f"  database: {_human_bytes(storage['database_bytes'])}")
    lines.append(f"  blobs: {_human_bytes(storage['blobs_bytes'])}")
    lines.append(f"  vector store: {_human_bytes(storage['vector_store_bytes'])}")
    for st, size in storage["plugins"].items():
        lines.append(f"  plugin {st}: {_human_bytes(size)}")
    return "\n".join(lines)


def _human_bytes(n: int) -> str:
    """Format a byte count as a compact human-readable string (e.g. ``1.5 MiB``)."""
    size = float(n)
    # Step up binary units until the value is small enough to read at a glance.
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            # Whole bytes need no decimal; larger units get one for readability.
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - loop always returns first


def _render_mapping(data: Any, *, indent: int) -> list[str]:
    """Recursively render a nested mapping/value as indented ``key: value`` lines."""
    pad = " " * indent
    # Non-mapping leaf: render on its own line.
    if not isinstance(data, dict):
        return [f"{pad}{data}"]
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            # Nested mapping: header line then the children one level deeper.
            lines.append(f"{pad}{key}:")
            lines.extend(_render_mapping(value, indent=indent + 2))
        else:
            lines.append(f"{pad}{key}: {value}")
    return lines


def run_stats(config: Config, *, source_type: str | None = None, as_json: bool = False) -> None:
    """Gather and print statistics (human-readable, or JSON with ``as_json``)."""
    stats = gather_stats(config, source_type=source_type)
    if as_json:
        print(json.dumps(stats, indent=2, sort_keys=True))
    else:
        print(_render(stats))
