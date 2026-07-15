"""Read the Microsoft GraphRAG output (parquet) for the viewer.

Tolerant to column-name drift across GraphRAG versions: entities may key on
`title` or `name`; provenance on `text_unit_ids`. Missing files -> empty lists.
"""

from __future__ import annotations

from pathlib import Path


def _first(row, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _as_list(v):
    if v is None:
        return []
    try:
        # numpy arrays / pandas lists
        return list(v)
    except TypeError:
        return [v]


class GraphRagStore:
    """Reads entities / relationships / community reports from output parquet."""

    def __init__(self, output_dir: str | Path):
        self.dir = Path(output_dir)

    def exists(self) -> bool:
        return self.dir.is_dir() and any(self.dir.glob("*.parquet"))

    def _read(self, name: str):
        path = self.dir / f"{name}.parquet"
        if not path.exists():
            return []
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")

    def entities(self) -> list[dict]:
        out = []
        for r in self._read("entities"):
            out.append({
                "name": _first(r, "title", "name", default=""),
                "type": _first(r, "type", default=""),
                "description": _first(r, "description", default=""),
                "text_unit_ids": _as_list(_first(r, "text_unit_ids")),
            })
        return out

    def entity(self, name: str) -> dict | None:
        for e in self.entities():
            if e["name"] == name:
                return e
        return None

    def relationships(self) -> list[dict]:
        out = []
        for r in self._read("relationships"):
            out.append({
                "source": _first(r, "source", default=""),
                "target": _first(r, "target", default=""),
                "description": _first(r, "description", default=""),
                "weight": _first(r, "weight", "combined_degree", default=""),
                "text_unit_ids": _as_list(_first(r, "text_unit_ids")),
            })
        return out

    def relationships_of(self, name: str) -> list[dict]:
        return [r for r in self.relationships()
                if r["source"] == name or r["target"] == name]

    def community_reports(self) -> list[dict]:
        out = []
        for r in self._read("community_reports"):
            out.append({
                "community": str(_first(r, "community", "id", default="")),
                "level": _first(r, "level", default=""),
                "title": _first(r, "title", default=""),
                "summary": _first(r, "summary", default=""),
                "full_content": _first(r, "full_content", "content", default=""),
            })
        return out

    def community(self, community_id: str) -> dict | None:
        for c in self.community_reports():
            if c["community"] == community_id:
                return c
        return None

    def _text_unit_map(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self._read("text_units"):
            tid = str(_first(r, "id", "human_readable_id", default=""))
            out[tid] = [str(d) for d in _as_list(_first(r, "document_ids"))]
        return out

    def provenance_docs(self, text_unit_ids) -> list[str]:
        """Map GraphRAG text_unit_ids back to our Document Store doc_ids."""
        m = self._text_unit_map()
        docs: set[str] = set()
        for t in text_unit_ids:
            docs.update(m.get(str(t), []))
        return sorted(docs)
