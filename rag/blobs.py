"""Content-addressed blob store on the filesystem (PLAN §3.1).

Large binary content (raw emails, translated Markdown, …) is stored as immutable
files named by the SHA-256 of their content, sharded into two levels of
subdirectories. The database only ever holds the hash, never the bytes. Because
identical content maps to the same path, writes deduplicate for free and reads
never need a lock (files are never mutated in place).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Length of each shard directory component taken from the hex digest. Two levels
# of 2 hex chars gives 256*256 buckets — plenty to keep any single directory
# small for personal-scale data.
_SHARD_LEN: int = 2


class BlobStore:
    """Immutable, content-addressed blob storage rooted at a directory."""

    def __init__(self, root: Path) -> None:
        """Create a store under ``root`` (created on demand on first write).

        ``root`` is the directory that will hold the sharded blob tree; it does
        not need to exist yet.
        """
        # Root directory of the sharded blob tree.
        self.root: Path = root

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

    def put(self, data: bytes) -> str:
        """Store ``data`` and return its content hash.

        Idempotent: storing identical bytes again is a cheap no-op (the target
        file already exists). The write is atomic (temp file + rename) so a crash
        mid-write can never leave a partial, wrongly-named blob.
        """
        digest = self.hash_bytes(data)
        target = self.path_for(digest)
        if target.exists():
            # Content already stored — dedup, nothing to write.
            return digest
        # Ensure the shard directories exist before writing.
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a unique temp file in the same directory, then atomically
        # rename into place so readers never observe a half-written blob.
        tmp = target.with_name(f".{digest}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return digest

    def get(self, digest: str) -> bytes:
        """Return the bytes stored under ``digest``.

        Raises :class:`FileNotFoundError` if no such blob exists.
        """
        return self.path_for(digest).read_bytes()

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
