"""Email extractor: IMAP fetch newer than the configured cutoff.

local_id = Message-ID (globally stable, cross-folder). Falls back to the IMAP
uid when a message has no Message-ID header.
"""

from __future__ import annotations

from typing import Iterable


class EmailExtractor:
    name = "email"

    def enabled(self, cfg) -> bool:
        return cfg.email.enabled

    def iter_documents(self, cfg) -> Iterable[tuple[str, str, dict]]:
        # Imported lazily so notes-only setups need not install the imap reader.
        from imap_tools import AND
        from llama_index.readers.imap import ImapReader

        reader = ImapReader(
            host=cfg.email.host,
            username=cfg.email.username,
            password=cfg.email.password,
        )
        docs = reader.load_data(
            folder=cfg.email.folder,
            search_criteria=AND(date_gte=cfg.email.since),
            metadata_names=["headers", "uid", "subject", "from_"],
        )
        for doc in docs:
            local_id, metadata = _local_id_and_metadata(doc, cfg.email.folder)
            yield local_id, doc.text, metadata


def _local_id_and_metadata(doc, folder: str) -> tuple[str, dict]:
    """Message-ID as local_id (uid fallback); drop the bulky headers blob."""
    metadata = dict(doc.metadata)
    headers = metadata.pop("headers", None)
    message_id = None
    # imap-tools exposes headers as a LazyHeaders mapping (not a dict); duck-type.
    get = getattr(headers, "get", None)
    if callable(get):
        raw = get("message-id") or get("Message-ID")
        if isinstance(raw, (list, tuple)) and raw:
            message_id = str(raw[0])
        elif isinstance(raw, str):
            message_id = raw

    metadata["folder"] = folder
    if message_id:
        return message_id.strip().strip("<>"), metadata
    if metadata.get("uid"):
        return f"uid:{folder}:{metadata['uid']}", metadata
    # No Message-ID and no uid: fall back to a content-derived id so re-runs dedup.
    import hashlib
    return "hash:" + hashlib.sha256((doc.text or "").encode()).hexdigest()[:16], metadata
