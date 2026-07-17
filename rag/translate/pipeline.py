"""Translation pipeline: selection, invariant enforcement, bookkeeping (PLAN §5.2).

Selects source documents, invokes the chosen (stateless) translator, enforces the
invariants translators are not trusted with, and writes translation sets/parts —
or, in inspection/dry-run modes, reports without touching the DB.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..blobs import BlobStore
from ..config import Config
from ..db import Database
from ..errors import RagError
from ..gitinfo import git_revision, is_clean
from ..interrupt import interrupt_requested
from .base import Document, Translation
from .discovery import load_translator
from .store import SourceDocument, TranslationStore

# Filename of the SQLite metadata database within the storage dir.
_DB_FILENAME: str = "metadata.db"
# Subdirectory (under the storage dir) holding the content-addressed blobs.
_BLOBS_SUBDIR: str = "blobs"


class TranslationError(RagError):
    """Raised when a translator violates a pipeline invariant."""


@dataclass(frozen=True)
class TranslateOptions:
    """Options controlling a translation run."""

    translator: str = "dispatch"
    """Name of the translator module to use (default: the dispatcher)."""

    source_type: str | None = None
    """Restrict to one source_type."""

    doc_id: str | None = None
    """Restrict to one document id."""

    limit: int | None = None
    """Max number of documents to process."""

    force: bool = False
    """Retranslate even documents whose active translation is up to date."""

    dry_run: bool = False
    """Translate but do not write to the DB (allowed on a dirty repo)."""

    inspect: bool = False
    """Print translations for review instead of storing them (dirty repo OK)."""

    verbose: bool = False
    """Print one progress line per translated document."""


def _validate(document: SourceDocument, translations: list[Translation]) -> None:
    """Enforce the invariants a translator is not trusted with (PLAN §5.2).

    Every translation's ``doc_id`` must equal the source's, and all ``sub_doc_id``s
    within the set must be distinct. Raises :class:`TranslationError` otherwise.
    """
    seen: set[tuple[str, ...]] = set()
    for translation in translations:
        # Identity is the pipeline's to guarantee, not the translator's.
        if translation.doc_id != document.doc_id:
            raise TranslationError(
                f"translator returned doc_id {translation.doc_id!r} for source "
                f"{document.doc_id!r}"
            )
        # No two parts in one set may share a sub_doc_id.
        key = tuple(translation.sub_doc_id)
        if key in seen:
            raise TranslationError(
                f"translator emitted duplicate sub_doc_id {translation.sub_doc_id!r} "
                f"for {document.doc_id!r}"
            )
        seen.add(key)


def _report(document: SourceDocument, translations: list[Translation]) -> None:
    """Print translations for a document to stdout (inspection mode)."""
    # A simple, greppable dump: one header per part, then its Markdown text.
    for translation in translations:
        sub = "/".join(translation.sub_doc_id) or "(root)"
        print(f"===== {document.source_type}:{document.doc_id} [{sub}] =====")
        print(translation.text)
        print()


def run_translation(
    config: Config,
    options: TranslateOptions,
    *,
    db: Database | None = None,
) -> int:
    """Run a translation pass; return the number of documents processed.

    Loads the selected translator, iterates the documents needing translation
    (per ``options``), enforces invariants, and writes each result as a
    translation set — unless ``inspect``/``dry_run`` is set. Storing requires a
    clean git tree (so the recorded ``git_rev`` is meaningful); inspection and
    dry-run do not.
    """
    storage = config.storage_dir
    translator = load_translator(options.translator)
    blobs = BlobStore(storage / _BLOBS_SUBDIR)

    # Git gating: only a real DB write needs a clean tree + a meaningful revision.
    store_to_db = not (options.dry_run or options.inspect)
    git_rev = ""
    if store_to_db:
        if not is_clean():
            raise TranslationError(
                "refusing to store translations: the working tree is dirty. "
                "Commit your changes, or use --dry-run / --inspect."
            )
        git_rev = git_revision()

    owns_db = db is None
    database = db if db is not None else Database.open(storage / _DB_FILENAME)
    try:
        store = TranslationStore(database.conn, blobs)
        processed = 0
        for source in store.select_documents(
            source_type=options.source_type,
            doc_id=options.doc_id,
            limit=options.limit,
            force=options.force,
        ):
            # Build the top-level Document (sub_doc_id is empty at this level).
            document = Document(
                doc_id=source.doc_id,
                sub_doc_id=[],
                content=source.content,
                stable_metadata=source.stable_metadata,
                transient_metadata=source.transient_metadata,
            )
            translations = translator.translate(document)
            _validate(source, translations)

            if options.verbose and not options.inspect:
                # One progress line per document (inspect mode already dumps them).
                sub_ids = ", ".join("/".join(t.sub_doc_id) or "(root)" for t in translations)
                prefix = "would translate" if options.dry_run else "translated"
                print(
                    f"{prefix} {source.source_type}:{source.doc_id} (v{source.version}) "
                    f"-> {len(translations)} part(s) [{sub_ids}] via {options.translator}",
                    file=sys.stderr,
                )

            if options.inspect:
                _report(source, translations)
            elif not options.dry_run:
                # Persist as one active translation set (deactivates the old one).
                store.write_set(
                    source_type=source.source_type,
                    doc_id=source.doc_id,
                    source_version=source.version,
                    translator_name=options.translator,
                    git_rev=git_rev,
                    parts=[(t.sub_doc_id, t.text, t.stable_metadata) for t in translations],
                )
            processed += 1
            # Checkpoint: a document is fully translated and (if storing) committed
            # here — a safe point to honour a graceful interrupt.
            if interrupt_requested():
                print(
                    f"interrupted after {processed} document(s).", file=sys.stderr
                )
                break
        return processed
    finally:
        if owns_db:
            database.close()
