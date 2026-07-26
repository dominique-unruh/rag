"""fsck: check the invariants our storage model assumes but does not enforce.

A read-only maintenance tool. The DB has some invariants enforced by SQLite (FK
cascades, unique indexes), but several are only *assumed* — they can be broken by
a crash between the two stores (SQLite vs. the vector store vs. the blob store), a
half-finished run, a manual edit, or a bug. This tool re-checks them and reports
every violation it finds.

**Extensible by design**: each invariant is one ``check_*`` function taking a
:class:`FsckContext` and returning a list of :class:`Problem`. Add a check by
writing such a function and appending ``(label, fn)`` to :data:`_CHECKS`. Each
extraction plugin is also asked to check its own private state via
:meth:`rag.extract.base.Extractor.fsck`.

The tool never writes; it opens the database read-only.

Further checks worth adding (things the model assumes that could still break):
- Orphan blobs: blob files on disk referenced by no document/part (GC candidates).
- Missing points: an active part embedded for a model (embeddings row present) but
  with zero points in that model's collection (the reverse of a dangling point).
- Active-set version sanity: an active set whose ``source_version`` exceeds the
  document's current ``version`` (impossible unless something wrote out of order).
- Part uniqueness: duplicate ``sub_doc_id`` within a set (guarded by an index, so
  a violation would mean index corruption).
- Embeddings/point set agreement: a point whose payload ``set_id`` is not the
  document's active set (stale points from a superseded set).
- Blob decode: every referenced blob actually decodes under its codec marker.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..blobs import BlobStore
from ..config import Config
from ..errors import VectorStoreBusyError, VectorStoreError
from ..fsck import FixMode, ask_fix, print_problem
from ..secrets import Secrets

# Subdirectory (under the storage dir) holding each plugin's private state dir —
# mirrors :data:`rag.extract.runner._PLUGIN_STATE_SUBDIR`.
_PLUGIN_STATE_SUBDIR: str = "plugin_state"

# Filenames/subdirs under the storage dir (mirroring the pipeline modules).
_DB_FILENAME: str = "metadata.db"
_BLOBS_SUBDIR: str = "blobs"


@dataclass
class Problem:
    """One invariant violation found by a check, optionally with a fix."""

    message: str
    """Human-readable description of what is wrong."""

    severity: str = "error"
    """``"error"`` (a real inconsistency) or ``"warning"`` (suspicious, not fatal)."""

    check: str = ""
    """The label of the check that produced it (filled in by :func:`run_fsck`)."""

    fix_description: str | None = None
    """What :attr:`apply_fix` will do — shown by :func:`~rag.fsck.ask_fix`."""

    apply_fix: "Callable[[], str | None] | None" = None
    """Repair action; returns an optional info line. ``None`` = not auto-fixable."""


@dataclass
class FsckContext:
    """Shared resources handed to every check.

    ``conn`` is ``None`` when there is no database yet (nothing extracted);
    ``vectors`` is ``None`` when the vector store could not be opened (in which
    case ``vectors_error`` says why), so point checks degrade to a warning rather
    than crashing the whole run.
    """

    config: Config
    """The loaded configuration."""

    storage_dir: Path
    """Root storage directory (DB, blobs, plugin state all live under it)."""

    conn: sqlite3.Connection | None
    """Read-only metadata DB connection, or ``None`` if the DB does not exist."""

    blobs: BlobStore
    """The content-addressed blob store (for reference checks)."""

    vectors: object | None = None
    """The opened :class:`~rag.embed.vectorstore.VectorStore`, or ``None``."""

    vectors_error: str | None = None
    """Why the vector store could not be opened (when ``vectors`` is ``None``)."""


def check_dangling_translations(ctx: FsckContext) -> list[Problem]:
    """Translation sets must reference an existing document (FK, ON DELETE CASCADE)."""
    if ctx.conn is None:
        return []
    problems: list[Problem] = []
    # A set whose (source_type, doc_id) has no row in documents is an orphan the
    # cascade should have removed.
    for row in ctx.conn.execute(
        "SELECT ts.id, ts.source_type, ts.doc_id FROM translation_sets ts "
        "WHERE NOT EXISTS (SELECT 1 FROM documents d "
        "WHERE d.source_type = ts.source_type AND d.doc_id = ts.doc_id)"
    ):
        problems.append(Problem(
            f"translation_set {row['id']} references missing document "
            f"{row['source_type']}:{row['doc_id']}"
        ))
    return problems


def check_dangling_parts(ctx: FsckContext) -> list[Problem]:
    """Translation parts must belong to an existing set (FK, ON DELETE CASCADE)."""
    if ctx.conn is None:
        return []
    problems: list[Problem] = []
    for row in ctx.conn.execute(
        "SELECT tp.id, tp.set_id FROM translation_parts tp "
        "WHERE NOT EXISTS (SELECT 1 FROM translation_sets ts WHERE ts.id = tp.set_id)"
    ):
        problems.append(Problem(
            f"translation_part {row['id']} references missing set {row['set_id']}"
        ))
    return problems


def check_dangling_embeddings(ctx: FsckContext) -> list[Problem]:
    """Embeddings rows must reference an existing part *and* document."""
    if ctx.conn is None:
        return []
    problems: list[Problem] = []
    for row in ctx.conn.execute(
        "SELECT e.part_id, e.model, e.source_type, e.doc_id, e.set_id FROM embeddings e "
        "WHERE NOT EXISTS (SELECT 1 FROM translation_parts tp WHERE tp.id = e.part_id) "
        "OR NOT EXISTS (SELECT 1 FROM translation_sets ts WHERE ts.id = e.set_id) "
        "OR NOT EXISTS (SELECT 1 FROM documents d "
        "WHERE d.source_type = e.source_type AND d.doc_id = e.doc_id)"
    ):
        problems.append(Problem(
            f"embeddings row (part {row['part_id']}, model {row['model']!r}, "
            f"{row['source_type']}:{row['doc_id']}, set {row['set_id']}) "
            f"references a missing part/set/document"
        ))
    return problems


def check_multiple_active_translations(ctx: FsckContext) -> list[Problem]:
    """At most one active translation set per document (guarded by idx_active_set)."""
    if ctx.conn is None:
        return []
    problems: list[Problem] = []
    for row in ctx.conn.execute(
        "SELECT source_type, doc_id, COUNT(*) AS n FROM translation_sets "
        "WHERE is_active = 1 GROUP BY source_type, doc_id HAVING n > 1"
    ):
        problems.append(Problem(
            f"document {row['source_type']}:{row['doc_id']} has {row['n']} active "
            f"translation sets (must be <= 1)"
        ))
    return problems


def check_dangling_blob_refs(ctx: FsckContext) -> list[Problem]:
    """Every blob_hash referenced by a document or part must exist in the blob store."""
    if ctx.conn is None:
        return []
    problems: list[Problem] = []
    # Track hashes already confirmed present so a blob shared by many rows is
    # stat-ed once (documents and part blobs never collide, but dedup is cheap).
    checked: set[str] = set()
    for table, kind in (("documents", "document"), ("translation_parts", "translation-part")):
        for row in ctx.conn.execute(f"SELECT DISTINCT blob_hash FROM {table}"):
            digest = row["blob_hash"]
            if not digest or digest in checked:
                continue
            checked.add(digest)
            if not ctx.blobs.exists(digest):
                problems.append(Problem(
                    f"{kind} blob {digest} is referenced but missing from the blob store"
                ))
    return problems


def check_dangling_points(ctx: FsckContext) -> list[Problem]:
    """Every vector-store point must map to a current embeddings row and document.

    A point is dangling if its payload ``part_id`` has no ``embeddings`` row for
    the collection's model, or its ``(source_type, doc_id)`` is not a live
    document — i.e. it is a stale point left after a re-embed or a delete.
    """
    if ctx.conn is None:
        return []
    if ctx.vectors is None:
        return [Problem(
            f"vector store unavailable; skipped point checks ({ctx.vectors_error})",
            severity="warning",
        )]
    from ..embed.models import load_models

    problems: list[Problem] = []
    # Live documents, to catch points whose document has gone.
    doc_keys = {
        (r["source_type"], r["doc_id"])
        for r in ctx.conn.execute("SELECT source_type, doc_id FROM documents")
    }
    for spec in load_models(ctx.config.raw).values():
        if not ctx.vectors.has_collection(spec.collection):  # type: ignore[attr-defined]
            continue
        # part_ids the DB says are embedded for this model (the collection).
        valid_parts = {
            int(r[0]) for r in ctx.conn.execute(
                "SELECT part_id FROM embeddings WHERE model = ?", (spec.id,)
            )
        }
        dangling_ids: list[object] = []
        examples: list[str] = []
        for point_id, payload in ctx.vectors.iter_points(spec.collection):  # type: ignore[attr-defined]
            part_id = payload.get("part_id")
            key = (payload.get("source_type"), payload.get("doc_id"))
            if not isinstance(part_id, int) or part_id not in valid_parts or key not in doc_keys:
                dangling_ids.append(point_id)
                if len(examples) < 5:
                    examples.append(str(point_id))
        if dangling_ids:
            collection = spec.collection
            # Fix: delete exactly the dangling points from this collection. Bind the
            # collection + id list into the closure so each problem fixes its own.
            def _delete(collection: str = collection, ids: list[object] = dangling_ids) -> str:
                ctx.vectors.delete_points(collection, ids)  # type: ignore[attr-defined]
                return f"deleted {len(ids)} point(s) from {collection!r}"

            problems.append(Problem(
                f"collection {collection!r} ({spec.id}): {len(dangling_ids)} dangling "
                f"point(s) with no backing embeddings row/document "
                f"(e.g. {', '.join(examples)})",
                fix_description=(
                    f"Fix: delete {len(dangling_ids)} dangling point(s) from collection {collection!r}"
                ),
                apply_fix=_delete,
            ))
    return problems


def check_stray_collections(ctx: FsckContext) -> list[Problem]:
    """Warn about vector-store collections that match no configured model."""
    if ctx.vectors is None:
        return []
    from ..embed.models import load_models

    known = {spec.collection for spec in load_models(ctx.config.raw).values()}
    problems: list[Problem] = []
    for collection in ctx.vectors.list_collections():  # type: ignore[attr-defined]
        if collection not in known:
            count = ctx.vectors.count(collection)  # type: ignore[attr-defined]
            problems.append(Problem(
                f"collection {collection!r} exists in the vector store but matches no "
                f"configured model ({count} point(s))",
                severity="warning",
            ))
    return problems


# The check registry: ``(label, function)``. Append here to add an invariant.
# Extractor self-checks are run separately (they print and fix themselves, via
# ``Extractor.fsck``), see :func:`_run_extractor_fsck`.
_CHECKS: list[tuple[str, Callable[[FsckContext], list[Problem]]]] = [
    ("dangling-translations", check_dangling_translations),
    ("dangling-parts", check_dangling_parts),
    ("dangling-embeddings", check_dangling_embeddings),
    ("multiple-active-translations", check_multiple_active_translations),
    ("dangling-blob-refs", check_dangling_blob_refs),
    ("dangling-points", check_dangling_points),
    ("stray-collections", check_stray_collections),
]


def _run_extractor_fsck(ctx: FsckContext, fix: FixMode) -> int:
    """Run each extractor's own ``fsck`` (which prints + fixes); return error count.

    Each plugin validates its private state dir, prints its own problems and
    applies fixes per ``fix``. A plugin that raises is reported but does not abort
    the tool.
    """
    from ..extract.discovery import discover_plugins

    errors = 0
    for source_type, plugin_cls in sorted(discover_plugins().items()):
        # Locate (do NOT create) the plugin's private state dir; skip if unused.
        state_dir = ctx.storage_dir / _PLUGIN_STATE_SUBDIR / source_type
        if not state_dir.is_dir():
            continue
        try:
            errors += plugin_cls().fsck(state_dir, ctx.config.plugin_config(source_type), fix)
        except Exception as exc:  # a plugin's fsck must never crash the tool
            print_problem(f"[{source_type}] fsck raised: {exc}")
            errors += 1
    return errors


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open the metadata database read-only (rows as :class:`sqlite3.Row`)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def run_fsck(config: Config, *, secrets: Secrets | None = None, fix: FixMode = "no") -> int:
    """Run every registered check + each extractor's fsck; return the error count.

    Opens the metadata DB read-only, the blob store, and (best-effort) the vector
    store, then runs each check in turn, **printing** every problem as it is found
    (tagged with its check). Extractor self-checks print and, per ``fix``, repair
    their own state. The return value is the number of ``error``-severity problems
    (warnings do not count), so the CLI can set a non-zero exit code.
    """
    storage = config.storage_dir
    resolver = secrets if secrets is not None else Secrets(allow_prompt=False)

    db_path = storage / _DB_FILENAME
    conn = _open_ro(db_path) if db_path.exists() else None
    blobs = BlobStore(storage / _BLOBS_SUBDIR)

    # Open the vector store best-effort: if it is locked/unavailable, point checks
    # degrade to a warning rather than aborting the whole run.
    vectors: object | None = None
    vectors_error: str | None = None
    try:
        from ..embed.vectorstore import open_vector_store

        vectors = open_vector_store(config, secrets=resolver)
    except (VectorStoreBusyError, VectorStoreError) as exc:
        vectors_error = str(exc)

    ctx = FsckContext(
        config=config,
        storage_dir=storage,
        conn=conn,
        blobs=blobs,
        vectors=vectors,
        vectors_error=vectors_error,
    )
    errors = 0
    warnings = 0
    try:
        # Central DB / vector-store checks: print each problem immediately, then
        # offer its fix (if any) under the chosen policy.
        for label, check in _CHECKS:
            for problem in check(ctx):
                print_problem(f"[{label}] {problem.message}", severity=problem.severity)
                if problem.severity == "error":
                    errors += 1
                else:
                    warnings += 1
                if problem.apply_fix is not None and ask_fix(fix, problem.fix_description or "Fix?"):
                    info = problem.apply_fix()
                    if info:
                        print(f"  {info}")
        # Per-extractor self-checks print (and fix) themselves.
        errors += _run_extractor_fsck(ctx, fix)
    finally:
        if vectors is not None:
            vectors.close()  # type: ignore[attr-defined]
        if conn is not None:
            conn.close()

    if errors == 0 and warnings == 0:
        print("fsck: no problems found.")
    else:
        print(f"\nfsck: {errors} error(s), {warnings} warning(s).")
    return errors
