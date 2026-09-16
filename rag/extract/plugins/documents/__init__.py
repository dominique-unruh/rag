"""Filesystem document extraction plugin (``source_type`` = ``documents``) — PLAN §4.4.

Extracts files from one or more configured on-disk folders. Each regular file
under a folder becomes a document with ``doc_id = "<folder-id>:<relative-path>"``
and a blob holding the file's raw bytes. Stable metadata (path, size, mime-type,
plus any sidecar fields) defines the content; transient metadata (folder-id and
the file mtime) tracks volatile state without bumping the version.

The plugin serves several folder kinds with no per-kind code:

* a *mirror* produced by the RWTH intranet scraper, where each downloaded file
  sits next to a ``<file>.metadata.json`` **sidecar** whose JSON is folded into
  the document's stable metadata (and whose ``content_type`` fixes the mime-type);
* an *Obsidian vault* — a tree of ``.md`` files with no sidecars; its hidden
  ``.obsidian/`` config dir is skipped by the default excludes.

Change detection is cheap by default: a file is re-read only when its (effective)
mtime differs from the one recorded on the last run. ``full=true`` forces a
re-read of every file (the framework's :meth:`Store.upsert` still decides whether
the bytes/metadata actually changed).

Each folder may set ``sourcelink`` to choose how a document's source link (the
URL :meth:`Extractor.source_url` returns) is generated: ``metadata`` (default —
the sidecar ``url``, falling back to a local ``file:`` URL), ``local`` (always a
``file:`` URL to the absolute path) or ``obsidian`` (an ``obsidian://`` URL that
opens the file in Obsidian, treating the folder ``path`` as the vault root). The
link is computed on demand from the live config (never baked in at ingest), so
changing the folder ``path`` or the mode takes effect immediately.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ....errors import ConfigError, PluginError
from ....interrupt import interrupt_requested
from ....mime import MimeDetector
from ....secrets import Secrets
from ....timestamps import format_timestamp
from ...base import Extractor, Store

# Default ceiling on a single file's size. Files above this are skipped with an
# error (never silently), since embedding a huge blob is almost always a mistake.
_DEFAULT_MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MiB

# Suffix identifying a per-file metadata sidecar (same convention as the RWTH
# intranet scraper: ``<file>.metadata.json`` next to ``<file>``).
_SIDECAR_SUFFIX: str = ".metadata.json"

# How a folder's per-document source link is generated (the folder's optional
# ``sourcelink`` config key). ``metadata`` extracts the link from the file's
# sidecar (``url`` field), falling back to ``local`` when the sidecar has none;
# ``local`` produces a ``file:`` URL to the file's absolute path; ``obsidian``
# produces an ``obsidian://`` URL that opens the file in Obsidian, treating the
# folder's configured ``path`` as the vault root. ``metadata`` is the default (it
# preserves the historical behaviour of reading a sidecar ``url``).
_SOURCELINK_MODES: tuple[str, ...] = ("metadata", "local", "obsidian")
_DEFAULT_SOURCELINK: str = "metadata"

# Basename glob patterns skipped by default: hidden files/dirs (``.*`` covers an
# Obsidian vault's ``.obsidian/``), editor backups and transient temp files. The
# config ``exclude`` list extends (never replaces) these.
_DEFAULT_EXCLUDES: tuple[str, ...] = (".*", "*~", "*.swp", "*.tmp", "*.part", "#*#")

# Every option key the documents extractor accepts. A key=value token outside
# this set is rejected so a typo fails loudly instead of being ignored.
# ``verbose`` is the reserved flag the CLI injects from ``-v``.
_USER_OPTIONS: tuple[str, ...] = ("folder", "full", "count")
_KNOWN_OPTIONS: frozenset[str] = frozenset(_USER_OPTIONS) | {"verbose"}


def _truthy(value: Any) -> bool:
    """Interpret a CLI option value as a boolean.

    Options arrive as strings (``full=true``) or, in tests, as real bools.
    Anything in the accepted true-set (case-insensitively) is ``True``; a bare
    missing option (``None``) is ``False``.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _folders_from_config(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the validated list of folder mappings from the plugin config.

    Requires a non-empty ``folders`` list, each entry a mapping with a non-empty
    ``folder-id`` and ``path``. Folder-ids must be unique (they prefix doc-ids),
    so a duplicate is a configuration error. An optional ``sourcelink`` must, if
    present, be one of :data:`_SOURCELINK_MODES`.
    """
    folders = config.get("folders")
    if not isinstance(folders, list) or not folders:
        raise ConfigError("documents plugin requires a non-empty 'folders' list in its config")
    seen: set[str] = set()
    for folder in folders:
        if not isinstance(folder, Mapping) or not folder.get("folder-id") or not folder.get("path"):
            raise ConfigError("each documents folder needs a mapping with a 'folder-id' and a 'path'")
        folder_id = str(folder["folder-id"])
        if folder_id in seen:
            raise ConfigError(f"duplicate documents folder-id {folder_id!r}")
        seen.add(folder_id)
        # An explicit sourcelink must name a known mode; a typo must fail loudly.
        sourcelink = folder.get("sourcelink")
        if sourcelink is not None and str(sourcelink).strip().lower() not in _SOURCELINK_MODES:
            raise ConfigError(
                f"documents folder {folder_id!r} has invalid sourcelink "
                f"{sourcelink!r}; expected one of {', '.join(_SOURCELINK_MODES)}"
            )
    return folders


def _folder_by_id(config: Mapping[str, Any], folder_id: str) -> Mapping[str, Any] | None:
    """Return the folder config entry with the given ``folder-id``, or ``None``.

    Used by :meth:`DocumentsExtractor.source_url` to resolve a document's folder
    (and thus its live root path + ``sourcelink`` mode) at query time. Tolerant of
    a missing/misshaped ``folders`` list — a link is best-effort, never fatal.
    """
    folders = config.get("folders")
    if not isinstance(folders, list):
        return None
    for folder in folders:
        if isinstance(folder, Mapping) and str(folder.get("folder-id")) == folder_id and folder.get("path"):
            return folder
    return None


def _sniff_mime(
    detector: MimeDetector, path: Path, data: bytes, sidecar: Mapping[str, Any] | None
) -> str:
    """Determine the mime type for a file: sidecar override, else content sniff.

    An explicit sidecar ``content_type`` wins verbatim; otherwise the shared
    :class:`MimeDetector` content-sniffs the bytes (libmagic), falling back to the
    file's extension only when magic is uninformative. Content sniffing is a hard
    requirement — the plugin never silently downgrades to extension-only detection.
    """
    # The source declared a type in its sidecar — trust it verbatim.
    if sidecar is not None:
        declared = sidecar.get("content_type")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    # Otherwise delegate to the shared detector (magic + extension fallback), the
    # same logic the archive translator uses for files it unpacks.
    return detector.detect(path.name, data)


def _is_excluded(name: str, patterns: Iterable[str]) -> bool:
    """Return whether a basename matches any of the exclusion glob patterns."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _load_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    """Read and parse a ``<file>.metadata.json`` sidecar, or return ``None``.

    Returns ``None`` when the sidecar is absent. A present-but-unreadable or
    malformed sidecar is a soft failure: a warning is printed and ``None`` is
    returned so the file is still ingested (just without the extra metadata).
    """
    if not sidecar_path.is_file():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: ignoring unreadable sidecar {sidecar_path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"warning: sidecar {sidecar_path} is not a JSON object; ignoring", file=sys.stderr)
        return None
    return data


def _iter_files(root: Path, excludes: tuple[str, ...]) -> Iterable[tuple[Path, str]]:
    """Yield ``(absolute_path, relative_posix_path)`` for every ingestible file.

    Walks ``root`` recursively in a deterministic (sorted) order. Symlinks (files
    and dirs) are skipped, excluded directories are pruned so their subtrees are
    never descended, and excluded files — as well as ``*.metadata.json`` sidecars
    (which are consumed with their partner file, never ingested on their own) —
    are omitted.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded and symlinked subdirectories in place so os.walk skips
        # them entirely; sort for a stable traversal order.
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            full = os.path.join(dirpath, dirname)
            if _is_excluded(dirname, excludes) or os.path.islink(full):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            # Sidecars are metadata for their partner file, not documents.
            if filename.endswith(_SIDECAR_SUFFIX):
                continue
            if _is_excluded(filename, excludes):
                continue
            abs_path = Path(dirpath) / filename
            if abs_path.is_symlink():
                continue
            rel_posix = abs_path.relative_to(root).as_posix()
            yield abs_path, rel_posix


def _effective_mtime(file_path: Path, sidecar_path: Path) -> str:
    """Return the canonical mtime for change detection: newest of file + sidecar.

    Taking the max means editing a sidecar (without touching the file) still
    retriggers ingestion. Rendered via the shared timestamp formatter so it is a
    stable, comparable UTC ISO 8601 string.
    """
    newest = file_path.stat().st_mtime
    if sidecar_path.is_file():
        newest = max(newest, sidecar_path.stat().st_mtime)
    return format_timestamp(datetime.fromtimestamp(newest, tz=timezone.utc))


def _source_link(mode: str, root: Path, rel_posix: str, metadata_url: str | None) -> str | None:
    """Build a document's source link for the folder's ``sourcelink`` mode.

    * ``local`` — a ``file:`` URL to the file's absolute path (percent-encoded).
    * ``obsidian`` — an ``obsidian://open`` URL that opens the file in Obsidian,
      taking ``root`` as the vault (its basename is the vault name) and the file's
      path relative to the root as the in-vault target.
    * ``metadata`` — ``metadata_url`` (the sidecar-derived ``url``) verbatim,
      falling back to a ``local`` ``file:`` URL when there is none.

    ``root`` (the folder's configured absolute path) and ``mode`` are read live
    from config at call time, so moving the storage or changing the mode takes
    effect immediately — nothing is baked in at ingest. Returns ``None`` only for
    an unrecognised ``mode`` (validation should prevent that).
    """
    if mode == "obsidian":
        # Obsidian's URI scheme: vault=<vault name>, file=<path within the vault>.
        # The vault name is the root directory's basename; the file is the path
        # relative to that root. Both components are percent-encoded (safe="" so
        # even slashes in the vault name, though unlikely, are escaped).
        vault = quote(root.name, safe="")
        return f"obsidian://open?vault={vault}&file={quote(rel_posix)}"
    # For both `local` and the `metadata` fallback we need the file: URL to the
    # absolute path. as_uri() requires an absolute path and percent-encodes it.
    file_url = (root / rel_posix).as_uri()
    if mode == "local":
        return file_url
    if mode == "metadata":
        # Prefer the metadata (sidecar) url; otherwise fall back to the local URL.
        if metadata_url is not None and metadata_url.strip():
            return metadata_url.strip()
        return file_url
    return None


class DocumentsExtractor(Extractor):
    """Extractor that syncs one or more on-disk folders into the store."""

    def source_url(
        self,
        doc_id: str,
        stable: Mapping[str, Any],
        transient: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> str | None:
        """Return the document's source link, honouring the folder's ``sourcelink``.

        Computed live from ``config`` (the documents plugin subtree), not baked in
        at ingest — so a changed folder ``path`` or ``sourcelink`` mode takes effect
        immediately. The document's folder-id (from transient ``folder``, or the
        ``doc_id`` prefix) selects the folder entry; its ``path`` is the root and
        its ``sourcelink`` the mode. The file's in-folder path comes from stable
        ``path`` (or the ``doc_id`` suffix), and the ``metadata`` mode's url from
        stable ``url``. Falls back to the stable ``url`` (else ``None``) if the
        folder is no longer configured.
        """
        # The doc_id is "<folder-id>:<relative-path>"; transient carries the same
        # folder-id explicitly. Prefer the explicit field, fall back to the prefix.
        folder_id = transient.get("folder")
        if not isinstance(folder_id, str) or not folder_id:
            folder_id = doc_id.split(":", 1)[0]
        rel_posix = stable.get("path")
        if not isinstance(rel_posix, str) or not rel_posix:
            rel_posix = doc_id.split(":", 1)[1] if ":" in doc_id else doc_id

        folder = _folder_by_id(config, folder_id)
        metadata_url = stable.get("url")
        metadata_url = metadata_url if isinstance(metadata_url, str) else None
        if folder is not None:
            root = Path(str(folder["path"])).expanduser()
            mode = str(folder.get("sourcelink") or _DEFAULT_SOURCELINK).strip().lower()
            link = _source_link(mode, root, rel_posix, metadata_url)
            if link is not None:
                return link
        # Folder gone from config (renamed/removed): best-effort sidecar url.
        return metadata_url if metadata_url else None

    def shortname(self, store: Store, doc_id: str, max_length: int | None = None) -> str:
        """Return a human-readable label: the sidecar title, else the file name.

        Falls back to the base class (the raw ``doc_id``) if no metadata is stored.
        """
        metadata = store.get_metadata(doc_id)
        if metadata:
            # Prefer an explicit sidecar title; otherwise the file's basename.
            title = metadata.get("title")
            path = metadata.get("path")
            label: str | None = None
            if isinstance(title, str) and title.strip():
                label = title.strip()
            elif isinstance(path, str) and path:
                label = path.rsplit("/", 1)[-1]
            if label is not None:
                if max_length is not None and len(label) > max_length:
                    return label[:max_length] + "…"
                return label
        return super().shortname(store, doc_id, max_length)

    def statistics(self, state_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
        """Report the configured folder ids (cheap; reads only the config)."""
        folders = config.get("folders")
        if not isinstance(folders, list):
            return {"folders_configured": 0}
        ids = [str(f["folder-id"]) for f in folders if isinstance(f, Mapping) and f.get("folder-id")]
        return {"folders_configured": len(ids), "folder_ids": ids}

    def run(
        self,
        store: Store,
        state_dir: Path,
        config: Mapping[str, Any],
        secrets: Secrets,
        **options: Any,
    ) -> None:
        """Reconcile the configured on-disk folders into ``store``.

        For each folder (optionally restricted by ``folder=``), walks its tree and
        upserts every ingestible file, skipping files whose mtime is unchanged
        since the last run unless ``full=true`` is given. After fully walking a
        folder it deletes documents whose files have vanished. ``count=N`` bounds
        the number of files read this run (shared across folders); reaching it, or
        a graceful interrupt, disables deletion detection for partially-walked
        folders so nothing is wrongly removed. An unrecognised option raises
        :class:`PluginError`.
        """
        # Reject unrecognised options up front (a typo like ``limit=1`` for
        # ``count=1`` would otherwise change behaviour silently).
        unknown = sorted(set(options) - _KNOWN_OPTIONS)
        if unknown:
            raise PluginError(
                f"unknown option(s): {', '.join(unknown)}. "
                f"Recognised options: {', '.join(_USER_OPTIONS)}."
            )
        # Fail fast if content sniffing is unavailable (never silently degrade).
        detector = MimeDetector()

        folders = _folders_from_config(config)
        verbose = _truthy(options.get("verbose"))
        force = _truthy(options.get("full"))
        only_folder = options.get("folder")
        # Shared budget across all folders (count=1 => one file total). None = ∞.
        remaining: int | None = int(options["count"]) if options.get("count") is not None else None

        # The size cap is a plugin-wide config value, not per folder.
        max_size = config.get("max_file_size")
        max_size = int(max_size) if max_size is not None else _DEFAULT_MAX_FILE_SIZE
        # Default excludes plus any user-configured additions.
        extra_excludes = config.get("exclude") or []
        if not isinstance(extra_excludes, list):
            raise ConfigError("documents 'exclude' must be a list of glob patterns")
        excludes = _DEFAULT_EXCLUDES + tuple(str(p) for p in extra_excludes)

        for folder in folders:
            folder_id = str(folder["folder-id"])
            if only_folder is not None and folder_id != only_folder:
                continue
            root = Path(str(folder["path"])).expanduser()
            if not root.is_dir():
                raise ConfigError(f"documents folder {folder_id!r} path is not a directory: {root}")

            if verbose:
                print(f"scanning folder {folder_id} ({root})", file=sys.stderr)

            # Ingest every file in this folder, tracking which doc-ids we saw and
            # whether we managed to walk the whole folder (needed before we may
            # delete vanished documents).
            seen_ids, completed, remaining = self._sync_folder(
                store=store, detector=detector, folder_id=folder_id, root=root,
                excludes=excludes, max_size=max_size, force=force,
                remaining=remaining, verbose=verbose,
            )

            # Deletion detection: only safe when the whole folder was walked
            # (a count-limited or interrupted pass has an incomplete `seen` set).
            if completed:
                self._delete_vanished(store, folder_id, seen_ids, verbose)
            elif verbose:
                print(f"  {folder_id}: partial pass — skipping deletion detection", file=sys.stderr)

            # A graceful interrupt or an exhausted budget stops the whole run.
            if interrupt_requested() or (remaining is not None and remaining <= 0):
                break

    def _sync_folder(
        self,
        *,
        store: Store,
        detector: MimeDetector,
        folder_id: str,
        root: Path,
        excludes: tuple[str, ...],
        max_size: int,
        force: bool,
        remaining: int | None,
        verbose: bool,
    ) -> tuple[set[str], bool, int | None]:
        """Ingest one folder; return ``(seen_doc_ids, completed, remaining_budget)``.

        ``completed`` is ``True`` only if the entire folder was walked without
        hitting the ``count`` budget or a graceful interrupt — the caller uses it
        to decide whether deletion detection is safe.
        """
        seen_ids: set[str] = set()
        completed = True
        for abs_path, rel_posix in _iter_files(root, excludes):
            # Stop cleanly on an exhausted budget or an interrupt; the folder is
            # only partially seen, so mark it incomplete.
            if remaining is not None and remaining <= 0:
                completed = False
                break
            if interrupt_requested():
                completed = False
                break

            doc_id = f"{folder_id}:{rel_posix}"
            seen_ids.add(doc_id)
            sidecar_path = abs_path.with_name(abs_path.name + _SIDECAR_SUFFIX)

            # Enforce the size cap loudly (never ingest a truncated/huge blob).
            try:
                size = abs_path.stat().st_size
            except OSError as exc:
                print(f"warning: cannot stat {abs_path}: {exc}", file=sys.stderr)
                continue
            if size > max_size:
                print(
                    f"error: {abs_path} is {size} bytes, over the {max_size}-byte "
                    "cap (max_file_size); skipping.",
                    file=sys.stderr,
                )
                continue

            mtime = _effective_mtime(abs_path, sidecar_path)
            # Cheap change detection: skip a re-read when the mtime is unchanged
            # (unless a full resync was requested).
            if not force:
                prior = store.get_transient(doc_id)
                if prior is not None and prior.get("mtime") == mtime:
                    continue

            try:
                content = abs_path.read_bytes()
            except OSError as exc:
                print(f"warning: cannot read {abs_path}: {exc}", file=sys.stderr)
                continue

            sidecar = _load_sidecar(sidecar_path)
            stable = self._build_stable(rel_posix, size, detector, abs_path, content, sidecar)
            transient = {"folder": folder_id, "mtime": mtime}

            result = store.upsert(doc_id, content, stable, transient=transient)
            if verbose:
                print(f"  {result.status.value} {doc_id} (v{result.version})", file=sys.stderr)
            # Only a file we actually read consumes the budget.
            if remaining is not None:
                remaining -= 1

        return seen_ids, completed, remaining

    def _build_stable(
        self,
        rel_posix: str,
        size: int,
        detector: MimeDetector,
        abs_path: Path,
        content: bytes,
        sidecar: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble the stable (content-defining) metadata for a file.

        Starts from the sidecar (provenance: title/url/facets/…), then sets the
        authoritative filesystem fields ``path``/``size`` and the resolved
        ``mime-type`` (which honours a sidecar ``content_type`` if present).
        """
        # Copy the sidecar first so our computed fields take precedence over any
        # like-named keys it might carry.
        stable: dict[str, Any] = dict(sidecar) if sidecar else {}
        stable["path"] = rel_posix
        stable["size"] = size
        stable["mime-type"] = _sniff_mime(detector, abs_path, content, sidecar)
        return stable

    def _delete_vanished(
        self, store: Store, folder_id: str, seen_ids: set[str], verbose: bool
    ) -> None:
        """Delete stored documents for this folder whose files no longer exist.

        Uses the transient ``folder`` filter so only this folder's documents are
        considered (never another folder's), then removes any id not seen in the
        just-completed walk.
        """
        stored = set(store.list_ids(transient={"folder": folder_id}))
        for doc_id in sorted(stored - seen_ids):
            store.delete(doc_id)
            if verbose:
                print(f"  deleted {doc_id} (vanished)", file=sys.stderr)
