"""Notes extractor: recursive *.md under the notes directory."""

from __future__ import annotations

from typing import Iterable

from llama_index.core import SimpleDirectoryReader


class NotesExtractor:
    name = "notes"

    def enabled(self, cfg) -> bool:
        return cfg.notes.enabled

    def iter_documents(self, cfg) -> Iterable[tuple[str, str, dict]]:
        reader = SimpleDirectoryReader(
            input_dir=cfg.notes.directory,
            required_exts=[".md"],
            recursive=True,
            filename_as_id=True,
        )
        for doc in reader.load_data():
            # filename_as_id -> doc.id_ is the absolute file path (stable local id).
            yield doc.id_, doc.text, dict(doc.metadata)
