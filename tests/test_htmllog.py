"""HtmlLog (embed table) + GraphLog (graph blocks) + source_label."""

from __future__ import annotations

from rag.htmllog import GraphLog, HtmlLog, source_label


class _FakeDoc:
    def __init__(self, metadata, doc_id):
        self.metadata = metadata
        self.doc_id = doc_id


class _N:
    def __init__(self, name):
        self.name = name
        self.id = name


class _R:
    def __init__(self, label):
        self.label = label


def test_htmllog_table_rows_append(tmp_path):
    path = tmp_path / "logs" / "embed.html"
    log = HtmlLog(path, "rag embed log")
    assert "rag embed log" in path.read_text()
    log.add("notes:/a.md", "a.md", "hello world")
    HtmlLog(path, "rag embed log").add("notes:/b.md", "b.md", "more")  # reopen, no dup header
    text = path.read_text()
    assert text.count("<!doctype html>") == 1
    assert "notes:/a.md" in text and "/b.md" in text and "hello world" in text
    assert text.count("<tr><td class=ts>") == 2


def test_graphlog_block_layout(tmp_path):
    path = tmp_path / "graph.html"
    log = GraphLog(path)
    log.add("notes:/a.md", "a.md", "source text here",
            relations=[(_N("Dominique"), _R("WORKS_AT"), _N("QIS")),
                       (_N("Alice"), _R("MET"), _N("Bob"))])
    log.add("notes:/b.md", "b.md", "text", relations=[])
    text = path.read_text()
    assert text.count("class=entry") == 2                 # one block per document
    assert "notes:/a.md" in text and "a.md" in text        # header line
    assert text.count("class=rel-row") == 3                # 2 relations + 1 "no relations"
    assert "WORKS_AT" in text and "MET" in text
    assert "no relations" in text                          # empty case
    assert "source text here" in text                      # foldable source text


def test_escapes_html(tmp_path):
    p = tmp_path / "g.html"
    GraphLog(p).add("notes:<x>", "<script>", "<b>b</b>", relations=[])
    text = p.read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text


def test_source_label_prefers_subject_then_filename():
    assert source_label(_FakeDoc({"subject": "Hi"}, "email:m@h")) == "Hi"
    assert source_label(_FakeDoc({"file_name": "w.md"}, "notes:/w.md")) == "w.md"
    assert source_label(_FakeDoc({}, "email:m@h")) == "email:m@h"  # falls back to doc_id
