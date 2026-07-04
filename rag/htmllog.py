"""Appendable HTML logs for the pipelines.

`HtmlLog` — embed pipeline: a table, one row per document (time, source, doc id,
foldable text).

`GraphLog` — graph pipeline: a block per document — a header line (time · doc id ·
source), then one row per relation created, then the source text (foldable). The
relations come from diffing the graph store around the insert (a back-query).

A header/style is written once per file; each document appends.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

_STYLE = """
body{font:13px/1.6 system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
h1{font-size:15px;margin:0;padding:12px 20px;background:#1a1a2e;color:#fff;position:sticky;top:0}
table{border-collapse:collapse;width:calc(100% - 40px);margin:16px 20px;background:#fff;
   box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid #f0f0f2;vertical-align:top}
th{background:#eef1fb;color:#334;font-size:12px}
.ts{color:#777;white-space:nowrap;font-size:12px}
.mono{font-family:ui-monospace,monospace;font-size:12px;word-break:break-all}
summary{cursor:pointer;color:#2657c7}
.txt{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;max-width:70ch;color:#333}
.entry{background:#fff;margin:14px 20px;border-radius:8px;padding:10px 14px;
   box-shadow:0 1px 3px rgba(0,0,0,.08)}
.meta{border-bottom:1px solid #f0f0f2;padding-bottom:6px;margin-bottom:6px}
.meta .src{color:#334}
.rel-row{font-family:ui-monospace,monospace;font-size:12px;padding:2px 0}
.rel{color:#888}
.empty{color:#999;font-style:italic}
"""


def source_label(sdoc) -> str:
    """Human descriptor: email subject / note filename, else the doc_id."""
    m = sdoc.metadata
    label = m.get("subject") or m.get("file_name") or m.get("file_path")
    return str(label) if label else sdoc.doc_id


def _node_name(node) -> str:
    return getattr(node, "name", None) or getattr(node, "id", "?")


def _text_details(text: str) -> str:
    text = text or ""
    summary = " ".join(text.split())[:60] or "(empty)"
    return (f"<details><summary>{html.escape(summary)}…</summary>"
            f"<div class=txt>{html.escape(text)}</div></details>")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _header(title: str, opener: str) -> str:
    return (f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
            f"<style>{_STYLE}</style><h1>{html.escape(title)}</h1>\n{opener}")


class HtmlLog:
    """Table log — one row per document (embed pipeline)."""

    def __init__(self, path: str | Path, title: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(_header(
                title,
                "<table><thead><tr><th>time</th><th>source document</th>"
                "<th>doc id</th><th>text</th></tr></thead><tbody>\n",
            ), encoding="utf-8")

    def add(self, doc_id: str, source: str, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"<tr><td class=ts>{_now()}</td>"
                f"<td class=mono>{html.escape(source)}</td>"
                f"<td class=mono>{html.escape(doc_id)}</td>"
                f"<td>{_text_details(text)}</td></tr>\n"
            )


class GraphLog:
    """Block log — per document: header, relation rows, foldable source text."""

    def __init__(self, path: str | Path, title: str = "rag graph log"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(_header(title, ""), encoding="utf-8")

    def add(self, doc_id: str, source: str, text: str, relations) -> None:
        relations = list(relations or [])
        if relations:
            rows = "".join(
                f"<div class=rel-row>{html.escape(_node_name(s))} "
                f"<span class=rel>-[{html.escape(r.label)}]-&gt;</span> "
                f"{html.escape(_node_name(t))}</div>"
                for s, r, t in relations
            )
        else:
            rows = "<div class=rel-row><span class=empty>no relations</span></div>"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"<div class=entry><div class=meta>"
                f"<span class=ts>{_now()}</span> · "
                f"<span class=mono>{html.escape(doc_id)}</span> · "
                f"<span class=src>{html.escape(source)}</span></div>"
                f"{rows}"
                f"<div>{_text_details(text)}</div></div>\n"
            )
