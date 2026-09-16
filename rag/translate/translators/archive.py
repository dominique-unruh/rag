"""The ``archive`` translator (PLAN §5.4).

Unpacks a container file (ZIP, TAR — including gzip/bzip2/xz-compressed tars —
and standalone gzip/bzip2/xz streams) in memory and routes **every contained
file back through the** ``dispatch`` **translator**, so archived documents of any
supported type (text, PDF, Office, even nested archives) are handled uniformly.

Nothing is written to disk: members are read into memory, so archive path
traversal is a non-issue. A per-member decompressed-size cap guards against
decompression bombs; oversized members are skipped with a warning rather than
aborting the run. Directory entries, symlinks and other non-regular members carry
no file content and are ignored.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from typing import Any

from ...mime import MimeDetector
from ..base import Document, Translation, Translator, make_translation

# Largest decompressed size accepted for a single archive member. A member that
# expands beyond this is skipped (defence against zip/decompression bombs), never
# silently truncated. Mirrors the documents extractor's per-file ceiling.
_MAX_MEMBER_BYTES: int = 100 * 1024 * 1024  # 100 MiB

# Magic-number prefixes for the standalone (non-tar) compression streams handled
# as single-file wrappers, mapped to the stdlib module used to decompress them.
_GZIP_MAGIC: bytes = b"\x1f\x8b"
_BZIP2_MAGIC: bytes = b"BZh"
_XZ_MAGIC: bytes = b"\xfd7zXZ\x00"

# Magic-number prefixes for the multi-file archive formats that need third-party
# libraries (py7zr / rarfile) rather than the stdlib. The RAR prefix is shared by
# both RAR4 (``\x00``) and RAR5 (``\x01\x00``) signatures.
_7Z_MAGIC: bytes = b"7z\xbc\xaf\x27\x1c"
_RAR_MAGIC: bytes = b"Rar!\x1a\x07"

# Compression-only extensions stripped from a standalone-stream member's name to
# recover the inner filename (``notes.txt.gz`` -> ``notes.txt``).
_STREAM_SUFFIXES: tuple[str, ...] = (".gz", ".bz2", ".xz", ".lzma", ".z")

# Timestamp rendering for the table-of-contents (minute precision, UTC-ish; the
# containers store no timezone, so the value is shown as-is).
_DATE_FMT: str = "%Y-%m-%d %H:%M"

# One row of the archive table of contents: (member name, uncompressed size in
# bytes or ``None`` if unknown, formatted modification time or "").
_TocRow = tuple[str, "int | None", str]


def _fmt_datetime(value: Any) -> str:
    """Format a ``datetime`` (or ``None``) as ``_DATE_FMT``; ``""`` if unusable."""
    if value is None:
        return ""
    try:
        return value.strftime(_DATE_FMT)
    except (AttributeError, ValueError):
        return ""


def _fmt_epoch(mtime: float) -> str:
    """Format a POSIX timestamp (tar's ``mtime``) as ``_DATE_FMT`` in UTC."""
    try:
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(_DATE_FMT)
    except (OSError, ValueError, OverflowError):
        return ""


def _fmt_date_tuple(parts: Any) -> str:
    """Format a ``(Y, M, D, h, m, s)`` tuple (zip/rar ``date_time``) as ``_DATE_FMT``.

    ZIP writes an all-zero tuple when it has no timestamp, which is not a valid
    date; a zero month or day is treated as "unknown" and rendered as ``""``.
    """
    try:
        year, month, day, hour, minute = (parts[0], parts[1], parts[2], parts[3], parts[4])
        if month == 0 or day == 0:
            return ""
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
    except (IndexError, TypeError, ValueError):
        return ""


def _toc_translation(document: Document, rows: list[_TocRow]) -> Translation:
    """Build the leading table-of-contents translation listing every member.

    Emitted *before* the member translations so a document whose members are all
    untranslatable (images, binaries, …) still yields useful indexable text: the
    file names, sizes and dates of what the archive holds. Its ``sub_doc_id`` is
    the parent's extended by ``"toc"``.
    """
    lines = ["| File | Size (bytes) | Modified |", "| --- | ---: | --- |"]
    for name, size, modified in rows:
        # Escape the table delimiter in file names; flatten nothing else.
        safe = name.replace("|", "\\|")
        size_text = "" if size is None else str(size)
        lines.append(f"| {safe} | {size_text} | {modified} |")
    text = "# Archive contents\n\n" + "\n".join(lines)
    return make_translation(document, text, sub_doc_id=[*document.sub_doc_id, "toc"])


def _require_py7zr() -> Any:
    """Import and return the ``py7zr`` module, or fail loudly with an install hint.

    Lazy so a missing py7zr breaks only 7z archives, not translator discovery or
    the stdlib-only container formats.
    """
    try:
        import py7zr  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PluginError(
            "the archive translator needs py7zr for 7z archives "
            "('pip install py7zr')."
        ) from exc
    return py7zr


def _require_rarfile() -> Any:
    """Import and return the ``rarfile`` module, or fail loudly with an install hint.

    ``rarfile`` is pure-Python metadata parsing but shells out to an external
    extractor (``unrar`` / ``bsdtar`` / ``unar``) for the compressed data; a
    missing *binary* surfaces later as :class:`~rarfile.RarCannotExec` when a
    member is read.
    """
    try:
        import rarfile  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PluginError(
            "the archive translator needs rarfile for RAR archives "
            "('pip install rarfile')."
        ) from exc
    return rarfile


def _dispatch_member(
    document: Document, name: str, data: bytes, detector: MimeDetector
) -> list[Translation]:
    """Build a synthetic document for one member and route it through dispatch.

    ``name`` is the member's archive-internal path; it extends the parent's
    ``sub_doc_id`` (keeping every part uniquely addressable and traceable back to
    its position inside nested archives). The member's mime type is content-sniffed
    via the shared :class:`MimeDetector`.
    """
    from .dispatch import translator as dispatch  # lazy: avoids an import cycle

    synthetic = Document(
        doc_id=document.doc_id,
        sub_doc_id=[*document.sub_doc_id, name],
        content=data,
        stable_metadata={
            "mime-type": detector.detect(name, data),
            "filename": name,
        },
    )
    # One corrupt member (e.g. a truncated PDF) must not abort the whole archive:
    # its sibling files and the table of contents are still worth indexing. Isolate
    # the failure to this member — warn and drop it. (Base/KeyboardInterrupt still
    # propagate so an interrupt is honoured.)
    try:
        return dispatch.translate(synthetic)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: untrusted member
        print(
            f"warning: skipping archive member {name!r} "
            f"({document.doc_id}): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return []


def _read_capped(fileobj: Any, name: str) -> bytes | None:
    """Read a decompressing file object, capping the result at ``_MAX_MEMBER_BYTES``.

    Returns the bytes, or ``None`` (with a warning) when the member expands past
    the cap — the guard against decompression bombs, since a compressed stream's
    real size is only known as it is read.
    """
    # Read one byte past the cap so exceeding it is detectable without holding an
    # unbounded amount of data in memory.
    data = fileobj.read(_MAX_MEMBER_BYTES + 1)
    if len(data) > _MAX_MEMBER_BYTES:
        print(
            f"warning: archive member {name!r} exceeds {_MAX_MEMBER_BYTES} bytes "
            f"decompressed; skipping",
            file=sys.stderr,
        )
        return None
    return data


def _translate_zip(document: Document, detector: MimeDetector) -> list[Translation]:
    """Unpack a ZIP archive: a table of contents plus each regular-file member."""
    rows: list[_TocRow] = []
    results: list[Translation] = []
    with zipfile.ZipFile(io.BytesIO(document.content)) as archive:
        for info in archive.infolist():
            # Directory entries (trailing "/") hold no file content.
            if info.is_dir():
                continue
            # Every file is listed in the TOC, even oversized/untranslatable ones.
            rows.append((info.filename, info.file_size, _fmt_date_tuple(info.date_time)))
            # Reject before reading: the header's uncompressed size is trusted
            # enough to skip an obvious bomb without decompressing it.
            if info.file_size > _MAX_MEMBER_BYTES:
                print(
                    f"warning: archive member {info.filename!r} exceeds "
                    f"{_MAX_MEMBER_BYTES} bytes decompressed; skipping",
                    file=sys.stderr,
                )
                continue
            with archive.open(info) as member:
                data = _read_capped(member, info.filename)
            if data is not None:
                results.extend(_dispatch_member(document, info.filename, data, detector))
    return [_toc_translation(document, rows), *results]


def _translate_tar(document: Document, detector: MimeDetector) -> list[Translation]:
    """Unpack a (optionally compressed) TAR: a table of contents plus each member."""
    rows: list[_TocRow] = []
    results: list[Translation] = []
    # mode "r:*" auto-detects the compression (plain/gzip/bzip2/xz).
    with tarfile.open(fileobj=io.BytesIO(document.content), mode="r:*") as archive:
        for member in archive:
            # Only regular files carry indexable bytes; skip dirs, symlinks, etc.
            if not member.isfile():
                continue
            rows.append((member.name, member.size, _fmt_epoch(member.mtime)))
            if member.size > _MAX_MEMBER_BYTES:
                print(
                    f"warning: archive member {member.name!r} exceeds "
                    f"{_MAX_MEMBER_BYTES} bytes decompressed; skipping",
                    file=sys.stderr,
                )
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                # Race/edge case: member vanished or is not extractable.
                continue
            with extracted:
                data = _read_capped(extracted, member.name)
            if data is not None:
                results.extend(_dispatch_member(document, member.name, data, detector))
    return [_toc_translation(document, rows), *results]


def _inner_name(document: Document) -> str:
    """Derive a standalone-stream member's name by stripping the compression suffix.

    Uses the source document's ``filename`` metadata (``notes.txt.gz`` ->
    ``notes.txt``); falls back to ``"content"`` when no usable name is known, so
    the inner file still gets dispatched (its type is content-sniffed anyway).
    """
    name = str(document.stable_metadata.get("filename") or "").rsplit("/", 1)[-1]
    if name.lower().endswith(_STREAM_SUFFIXES):
        name = name[: name.rfind(".")]
    return name or "content"


def _translate_stream(
    document: Document, decompressor: Any, detector: MimeDetector
) -> list[Translation]:
    """Decompress a standalone gzip/bzip2/xz stream and dispatch the single file."""
    with decompressor(io.BytesIO(document.content)) as stream:
        data = _read_capped(stream, _inner_name(document))
    if data is None:
        return []
    return _dispatch_member(document, _inner_name(document), data, detector)


def _translate_7z(document: Document, detector: MimeDetector) -> list[Translation]:
    """Unpack a 7-Zip archive (via ``py7zr``) and dispatch each file member.

    ``py7zr`` extracts to writer objects rather than a stream API, so an
    in-memory ``BytesIOFactory`` collects each member's bytes without touching
    disk. Member sizes are read from the header first, so oversized entries are
    skipped before they are decompressed.
    """
    py7zr = _require_py7zr()
    from py7zr.io import BytesIOFactory  # lazy: only when a 7z is actually seen

    rows: list[_TocRow] = []
    results: list[Translation] = []
    with py7zr.SevenZipFile(io.BytesIO(document.content), "r") as archive:
        # Choose the file members under the cap; directories carry no content.
        targets: list[str] = []
        for info in archive.list():
            if info.is_directory:
                continue
            rows.append((info.filename, info.uncompressed, _fmt_datetime(info.creationtime)))
            if info.uncompressed > _MAX_MEMBER_BYTES:
                print(
                    f"warning: archive member {info.filename!r} exceeds "
                    f"{_MAX_MEMBER_BYTES} bytes decompressed; skipping",
                    file=sys.stderr,
                )
                continue
            targets.append(info.filename)
        if targets:
            # Decompress the selected members into memory in one pass.
            factory = BytesIOFactory(_MAX_MEMBER_BYTES)
            archive.extract(factory=factory, targets=targets)
            for name in targets:
                stream = factory.get(name)
                if stream is None:
                    continue
                results.extend(_dispatch_member(document, name, stream.read(), detector))
    return [_toc_translation(document, rows), *results]


def _translate_rar(document: Document, detector: MimeDetector) -> list[Translation]:
    """Unpack a RAR archive (via ``rarfile``) and dispatch each file member.

    ``rarfile`` needs an external extractor binary (``unrar`` / ``bsdtar`` /
    ``unar``) for the compressed data; its absence is turned into a loud
    :class:`PluginError` with an install hint rather than a bare backend error.
    """
    rarfile = _require_rarfile()
    rows: list[_TocRow] = []
    results: list[Translation] = []
    try:
        with rarfile.RarFile(io.BytesIO(document.content)) as archive:
            for info in archive.infolist():
                # Directory entries hold no file content.
                if info.is_dir():
                    continue
                rows.append((info.filename, info.file_size, _fmt_date_tuple(info.date_time)))
                # The header's uncompressed size gates an obvious bomb up front.
                if info.file_size > _MAX_MEMBER_BYTES:
                    print(
                        f"warning: archive member {info.filename!r} exceeds "
                        f"{_MAX_MEMBER_BYTES} bytes decompressed; skipping",
                        file=sys.stderr,
                    )
                    continue
                with archive.open(info) as member:
                    data = _read_capped(member, info.filename)
                if data is not None:
                    results.extend(_dispatch_member(document, info.filename, data, detector))
    except rarfile.RarCannotExec as exc:
        # rarfile parsed the archive but no extractor binary is installed.
        raise PluginError(
            "reading RAR archives needs an external extractor; install one of "
            "'unrar', 'bsdtar' (libarchive-tools) or 'unar'."
        ) from exc
    return [_toc_translation(document, rows), *results]


class Archive(Translator):
    """Unpacks a container file and dispatches every contained file."""

    def translate(self, document: Document) -> list[Translation]:
        """Route each member of the archive back through ``dispatch``.

        The container format is detected from the content itself (not the declared
        mime type), so a mislabelled archive is still handled: ZIP first, then the
        TAR family (including compressed tars), then standalone compression
        streams. Content that matches none is left untranslated (empty result).
        """
        # Resolve libmagic up front (fails loudly if unavailable); a fresh detector
        # per call keeps the translator stateless and thread-safe.
        detector = MimeDetector()
        content = document.content
        # ZIP is checked first: its central-directory signature is unambiguous.
        if zipfile.is_zipfile(io.BytesIO(content)):
            return _translate_zip(document, detector)
        # TAR (plain or compressed) before the standalone streams, so a ``.tar.gz``
        # is unpacked into its members rather than treated as one gzip blob.
        try:
            is_tar = tarfile.is_tarfile(io.BytesIO(content))
        except (tarfile.TarError, EOFError):
            is_tar = False
        if is_tar:
            return _translate_tar(document, detector)
        # Standalone single-file compression streams, dispatched by magic number.
        # The module-level ``open`` functions accept the in-memory file object as
        # their first argument (unlike ``gzip.GzipFile``, whose first arg is a path).
        if content.startswith(_GZIP_MAGIC):
            return _translate_stream(document, gzip.open, detector)
        if content.startswith(_BZIP2_MAGIC):
            return _translate_stream(document, bz2.open, detector)
        if content.startswith(_XZ_MAGIC):
            return _translate_stream(document, lzma.open, detector)
        # Multi-file formats needing third-party libraries.
        if content.startswith(_7Z_MAGIC):
            return _translate_7z(document, detector)
        if content.startswith(_RAR_MAGIC):
            return _translate_rar(document, detector)
        # Not a container format we recognise; nothing to unpack.
        print(
            f"warning: archive translator got unrecognised container "
            f"({document.doc_id}:{'/'.join(document.sub_doc_id) or 'root'})",
            file=sys.stderr,
        )
        return []


# Module-level instance selected by --translator archive (and by dispatch).
translator: Archive = Archive()
