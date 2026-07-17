"""Content-addressed blob store on the filesystem (PLAN §3.1).

Large binary content (raw emails, translated Markdown, …) is stored as immutable
files named by the SHA-256 of their content, sharded into two levels of
subdirectories. The database only ever holds the hash, never the bytes. Because
identical content maps to the same path, writes deduplicate for free and reads
never need a lock (files are never mutated in place).

**On-disk encoding.** Each stored file is a **1-byte codec marker** followed by
the (possibly compressed) payload. The marker selects how the payload is decoded
back to the original content — ``0`` = raw, ``1`` = zstd — leaving up to 256 codec
slots for future formats. Crucially the file *name* (the content address) is the
SHA-256 of the **decoded** content, so the marker/codec is a pure storage detail:
dedup and immutability are unaffected, and re-encoding a blob never changes its
id. On write we keep whichever of raw/zstd is smaller (already-compressed
attachments do not shrink, so we never grow them beyond the 1-byte marker).

**Format migration.** Blobs written by an earlier, marker-less version of the
store are plain raw bytes with no marker. A one-time :meth:`BlobStore.migrate`
pass rewrites every such blob into the marker format and records the new format
version in a small sentinel file, so it runs once and is thereafter a cheap no-op.
Because that migration converts everything up front, the normal read path assumes
the marker format and carries no legacy-format handling.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import zstandard

from .errors import RagError

# Length of each shard directory component taken from the hex digest. Two levels
# of 2 hex chars gives 256*256 buckets — plenty to keep any single directory
# small for personal-scale data.
_SHARD_LEN: int = 2

# Codec markers: the single leading byte of every stored file. Room for 256
# codecs; add a new format by assigning an unused marker and handling it in
# ``_encode``/``_decode``.
_MARKER_RAW: int = 0
"""Payload is the content verbatim (no compression)."""
_MARKER_ZSTD: int = 1
"""Payload is a zstd frame of the content."""

# zstd compression level for new blobs. High (slow-ish) compression: the store is
# effectively write-once and zstd decompresses fast regardless. See PLAN
# (blob compression) for the measured rationale (~47% on-disk on real mail).
_ZSTD_LEVEL: int = 19

# Current on-disk format version. 1 = the marker-prefixed encoding introduced
# with zstd compression. A store at a lower/absent version is migrated up.
_CURRENT_FORMAT: int = 1

# Sentinel file (at the store root) recording the format version, so the one-time
# migration is skipped once complete.
_FORMAT_FILENAME: str = ".format_version"

# A blob filename is exactly the 64-char lowercase hex SHA-256 digest; used to
# pick real blobs out of the tree (skipping the sentinel and any temp files).
_DIGEST_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")


class BlobCorruptionError(RagError):
    """A stored blob could not be decoded to content matching its digest."""


class BlobStore:
    """Immutable, content-addressed blob storage rooted at a directory."""

    def __init__(self, root: Path) -> None:
        """Create a store under ``root`` (created on demand on first write).

        ``root`` is the directory that will hold the sharded blob tree; it does
        not need to exist yet.
        """
        # Root directory of the sharded blob tree.
        self.root: Path = root
        # Reused zstd (de)compressor contexts. ``write_content_size`` embeds the
        # decompressed size in the frame so ``decompress`` needs no size hint.
        self._compressor: zstandard.ZstdCompressor = zstandard.ZstdCompressor(
            level=_ZSTD_LEVEL, write_content_size=True
        )
        self._decompressor: zstandard.ZstdDecompressor = zstandard.ZstdDecompressor()

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Return the hex SHA-256 of ``data`` — the blob's content address."""
        return hashlib.sha256(data).hexdigest()

    def path_for(self, digest: str) -> Path:
        """Return the on-disk path for a given content ``digest``.

        The path is ``root/<aa>/<bb>/<digest>`` where ``aa``/``bb`` are the first
        two 2-char shards of the digest. Pure function: does not touch disk.
        """
        # Split off two shard components, keep the full digest as the filename so
        # the path is unambiguous and self-describing.
        first = digest[:_SHARD_LEN]
        second = digest[_SHARD_LEN : 2 * _SHARD_LEN]
        return self.root / first / second / digest

    def _encode(self, data: bytes) -> bytes:
        """Return the on-disk bytes for ``data``: 1 marker byte + smallest payload.

        Compresses with zstd and keeps it only when it actually beats the raw
        bytes, so incompressible content (already-compressed attachments) is stored
        raw and never grows beyond the single marker byte.
        """
        # Default to raw; adopt the compressed form only if it is strictly smaller.
        marker, payload = _MARKER_RAW, data
        compressed = self._compressor.compress(data)
        if len(compressed) < len(payload):
            marker, payload = _MARKER_ZSTD, compressed
        return bytes([marker]) + payload

    def _decode(self, blob: bytes) -> bytes | None:
        """Decode a marker-prefixed stored ``blob`` back to content, or ``None``.

        Returns ``None`` (rather than raising) when the leading byte is not a known
        codec or the payload fails to decompress — signalling a corrupt or
        unmigrated file to the caller.
        """
        if not blob:
            # An empty file carries no marker byte; it cannot be valid new-format.
            return None
        marker, payload = blob[0], blob[1:]
        if marker == _MARKER_RAW:
            return payload
        if marker == _MARKER_ZSTD:
            try:
                return self._decompressor.decompress(payload)
            except zstandard.ZstdError:
                # Payload is not a valid zstd frame.
                return None
        # Unknown codec marker.
        return None

    def put(self, data: bytes) -> str:
        """Store ``data`` and return its content hash.

        The hash is over the **uncompressed** ``data`` (the content address);
        storing identical bytes again is a cheap no-op regardless of how the
        existing copy was encoded. The write is atomic (temp file + rename) so a
        crash mid-write can never leave a partial, wrongly-named blob.
        """
        # Address by the plaintext content, not the encoded form.
        digest = self.hash_bytes(data)
        target = self.path_for(digest)
        if target.exists():
            # Content already stored (in whatever encoding) — dedup, nothing to do.
            return digest
        # Encode (marker + smallest payload) and write atomically.
        self._write_atomic(target, self._encode(data))
        return digest

    def _write_atomic(self, target: Path, encoded: bytes) -> None:
        """Write ``encoded`` to ``target`` via a same-dir temp file + rename.

        Ensures readers never observe a half-written blob and a crash can never
        leave a partial file under the final (hash) name.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name in the same directory so the rename is atomic (same FS).
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, target)

    def get(self, digest: str) -> bytes:
        """Return the bytes stored under ``digest``.

        Decodes the marker-prefixed encoding written by :meth:`put` (all blobs are
        in this format once :meth:`migrate` has run). Raises
        :class:`FileNotFoundError` if no such blob exists, or
        :class:`BlobCorruptionError` if the file cannot be decoded.
        """
        blob = self.path_for(digest).read_bytes()
        decoded = self._decode(blob)
        if decoded is None:
            # Unknown marker / bad frame: corruption, or a store that predates the
            # marker format and has not been migrated yet.
            raise BlobCorruptionError(
                f"blob {digest} has an unknown or invalid codec marker "
                "(is the store migrated?)"
            )
        return decoded

    def exists(self, digest: str) -> bool:
        """Return whether a blob with this ``digest`` is present."""
        return self.path_for(digest).exists()

    def delete(self, digest: str) -> bool:
        """Delete the blob for ``digest`` (used only by GC).

        Returns whether a file was actually removed. Empty shard directories are
        left in place; they are harmless and cheap.
        """
        target = self.path_for(digest)
        try:
            target.unlink()
        except FileNotFoundError:
            # Already gone — treat as a no-op so GC is idempotent.
            return False
        return True

    # --- one-time format migration --------------------------------------

    def _format_version(self) -> int:
        """Return the recorded on-disk format version (0 if never recorded)."""
        sentinel = self.root / _FORMAT_FILENAME
        try:
            data = json.loads(sentinel.read_text())
        except (FileNotFoundError, ValueError):
            # No sentinel (fresh or pre-migration store) → treat as version 0.
            return 0
        # Be defensive about a hand-edited/garbled sentinel.
        version = data.get("format_version") if isinstance(data, dict) else None
        return int(version) if isinstance(version, int) else 0

    def _set_format_version(self, version: int) -> None:
        """Record the on-disk format version in the sentinel file (atomically)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_atomic(
            self.root / _FORMAT_FILENAME,
            json.dumps({"format_version": version}).encode("utf-8"),
        )

    def _iter_blob_files(self) -> "list[Path]":
        """Return the paths of all real blob files under the store root.

        Filters to files whose name is a valid SHA-256 digest, so the sentinel and
        any leftover ``.tmp`` files are ignored.
        """
        if not self.root.is_dir():
            return []
        return [
            p for p in self.root.rglob("*")
            if p.is_file() and not p.is_symlink() and _DIGEST_RE.match(p.name)
        ]

    def migrate(self) -> int:
        """Bring the store up to the current on-disk format; return blobs rewritten.

        One-time, idempotent upgrade: blobs written by the pre-marker store are
        plain raw content, so each is re-encoded into the marker format (compressing
        where that helps) via an atomic rewrite. Blobs already in the marker format
        are left untouched. On completion the format version is recorded so
        subsequent calls short-circuit to a cheap no-op.

        Not safe against a concurrent writer — call it while nothing else touches
        the store. Aborts (raising :class:`BlobCorruptionError`) if a legacy blob's
        bytes do not match its own hash, rather than rewriting corrupt data.
        """
        # Already current: nothing to do (the common, post-migration case).
        if self._format_version() >= _CURRENT_FORMAT:
            return 0
        rewritten = 0
        for path in self._iter_blob_files():
            digest = path.name
            blob = path.read_bytes()
            # If it already decodes to its own content, it is new-format — skip.
            decoded = self._decode(blob)
            if decoded is not None and self.hash_bytes(decoded) == digest:
                continue
            # Otherwise it must be a legacy marker-less raw blob: its raw bytes are
            # the content. Verify against the address before touching it.
            if self.hash_bytes(blob) != digest:
                raise BlobCorruptionError(
                    f"blob {digest} is neither valid new-format nor matching legacy raw"
                )
            # Re-encode the raw content into the marker format, in place.
            self._write_atomic(path, self._encode(blob))
            rewritten += 1
        # Record completion so we never sweep the tree again.
        self._set_format_version(_CURRENT_FORMAT)
        return rewritten
