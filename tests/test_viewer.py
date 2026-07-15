"""viewer: routes over the graph-store interface + Document Store (fakes)."""

from __future__ import annotations

import os

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

from rag import config, viewer
from rag.documents import DocumentStore


class FakeGraph:
    """Minimal PropertyGraphStore: get() + get_triplets() over fixed data."""

    def __init__(self, nodes, triplets):
        self._nodes = nodes
        self._triplets = triplets

    def get(self, properties=None, ids=None):
        if ids is not None:
            return [n for n in self._nodes if n.id in ids]
        return list(self._nodes)

    def get_triplets(self, **kwargs):
        return list(self._triplets)

    def get_rel_map(self, nodes, depth=2, **kwargs):
        return list(self._triplets)

    def structured_query(self, query, param_map=None):
        # emulate Kùzu returning the edge with its stored properties (provenance)
        p = param_map or {}
        for s, r, t in self._triplets:
            if s.id == p.get("s") and t.id == p.get("t") and r.label == p.get("l"):
                props = dict(getattr(r, "properties", {}) or {})
                return [{"r": {"label": r.label, "_src": {}, "_dst": {}, **props}}]
        return []


@pytest.fixture
def seeded(project, tmp_path):
    """Fake graph (Dominique-WORKS_AT->QIS) + a notes document in the Doc Store."""
    root, _ = project
    cfg = config.load(root)

    notes_path = os.path.join(cfg.notes.directory, "meetings", "w.md")
    doc_id = f"notes:{notes_path}"

    chunk = ChunkNode(text="Dominique works at QIS in Berlin.",
                      properties={"file_path": notes_path})
    cid = chunk.id
    dom = EntityNode(name="Dominique", label="entity",
                     properties={"triplet_source_id": cid, "file_path": notes_path})
    qis = EntityNode(name="QIS", label="entity", properties={"triplet_source_id": cid})
    triplets = [(dom, Relation(label="WORKS_AT", source_id="Dominique",
                               target_id="QIS", properties={"triplet_source_id": cid}), qis)]

    docs = DocumentStore(tmp_path / "d.db")
    docs.upsert("notes", notes_path,
                "# Meeting\nDominique works at QIS in Berlin.\n",
                {"file_path": notes_path, "file_name": "w.md",
                 "_node_content": '{"id_":"c1","metadata":{"x":1}}'})

    viewer._CFG = cfg
    viewer._GRAPH = FakeGraph([chunk, dom, qis], triplets)
    viewer._DOCS = docs
    return viewer.app.test_client(), cid, doc_id, notes_path


def _text(resp):
    return resp.get_data(as_text=True)


def test_index_counts(seeded):
    client, _, doc_id, _ = seeded
    body = _text(client.get("/"))
    assert "2 entities · 1 relations · 1 documents" in body
    assert "(0 embedded, 0 graph-extracted)" in body  # ledgers empty in fixture
    assert "Dominique" in body


def test_entity_page_relations_and_links(seeded):
    client, _, _, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "entity", "id": "Dominique"}))
    assert 'class="kind entity"' in body and "object ·" not in body
    assert "WORKS_AT" in body and "id=QIS" in body


def test_relation_page_by_id(seeded):
    client, _, _, _ = seeded
    from llama_index.core.graph_stores.types import Relation
    rid = viewer._rel_id(Relation(label="WORKS_AT", source_id="Dominique", target_id="QIS"))
    body = _text(client.get("/item", query_string={"kind": "relation", "id": rid}))
    assert 'class="kind relation"' in body
    assert "id=Dominique" in body and "id=QIS" in body


def test_relation_page_shows_provenance(seeded):
    client, cid, _, _ = seeded
    from llama_index.core.graph_stores.types import Relation
    rid = viewer._rel_id(Relation(label="WORKS_AT", source_id="Dominique", target_id="QIS"))
    body = _text(client.get("/item", query_string={"kind": "relation", "id": rid}))
    # relation carries triplet_source_id -> fetched via structured_query, shown + linked
    assert "triplet_source_id" in body
    assert cid in body


def test_rel_id_roundtrip(seeded):
    client, _, _, _ = seeded
    from llama_index.core.graph_stores.types import Relation
    r = Relation(label="Is", source_id="Christiane hohnl", target_id="Praxismanagerin")
    rid = viewer._rel_id(r)
    graph = viewer._graph()
    got = viewer._rel_from_id(graph, rid)  # not in fake graph -> None, but decodes ok
    assert got is None  # fake graph lacks it; decode path exercised
    # the relation that IS in the graph round-trips
    real = viewer._rel_id(Relation(label="WORKS_AT", source_id="Dominique", target_id="QIS"))
    assert viewer._rel_from_id(graph, real).label == "WORKS_AT"


def test_chunk_page_markdown_and_entities(seeded):
    client, cid, _, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "chunk", "id": cid}))
    assert 'class="kind chunk"' in body
    assert "Dominique" in body  # entity extracted here


def test_document_page_from_docstore(seeded):
    client, _, doc_id, notes_path = seeded
    body = _text(client.get("/item", query_string={"kind": "document", "id": doc_id}))
    assert 'class="kind document"' in body
    assert "<h1>Meeting</h1>" in body            # markdown text rendered
    assert "file_name" in body                    # metadata table
    # local_id (path) -> obsidian link, not the "notes:" prefixed id
    assert "obsidian://open?vault=notes&amp;file=meetings%2Fw" in body


def test_missing_item_404(seeded):
    client, _, _, _ = seeded
    r = client.get("/item", query_string={"kind": "entity", "id": "Nobody"})
    assert r.status_code == 404


def test_search_text_and_id_jump(seeded):
    client, _, doc_id, _ = seeded
    assert "Graph matches" in _text(client.get("/search", query_string={"q": "Domin"}))
    # exact entity id jumps to the item page
    body = _text(client.get("/search", query_string={"q": "Dominique"}))
    assert "Relations (" in body
    # exact document id jumps to the document page
    dbody = _text(client.get("/search", query_string={"q": doc_id}))
    assert "Metadata" in dbody


def test_search_semantic_unavailable_graceful(seeded):
    client, _, _, _ = seeded
    body = _text(client.get("/search", query_string={"q": "x", "semantic": "1"}))
    assert "Semantic matches" in body and "unavailable" in body


def test_property_tooltip_and_json_tree(seeded):
    client, cid, _, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "chunk", "id": cid}))
    assert "class=pinfo" in body                  # ⓘ on described property
    assert "class=jt" in body or "file_path" in body


def test_history_letter_badges(seeded):
    client, _, _, _ = seeded
    body = _text(client.get("/"))
    assert "LETTER" in body and "relation:'R'" in body
    assert "rag_view_history" in body


# --- pure-function tests (no request) -------------------------------------

def test_path_translation(seeded):
    _c, _, _, notes_path = seeded
    assert viewer._translate_path(notes_path).startswith("obsidian://open?vault=notes")
    mid = "abc+def@mail.example.org"
    assert viewer._translate_path(mid) == "mid:abc+def@mail.example.org"  # raw, not encoded
    assert viewer._translate_path("/etc/passwd") is None


def test_json_tree_and_try_json():
    assert viewer._try_json("not json") is None
    assert viewer._try_json('{"k":1}') == {"k": 1}
    out = viewer._json_tree({"a": [1, True, None]})
    assert "<details class=jt" in out and "<span class=jkey>a</span>" in out


def test_render_md():
    assert "<strong>b</strong>" in viewer._render_md("**b**")


def test_idle_release_closes_when_idle():
    class FakeDocs:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    docs = FakeDocs()
    viewer._GRAPH = object()
    viewer._GRAPH_DB = None
    viewer._DOCS = docs
    # recent access -> kept
    viewer._last_access = viewer.time.monotonic()
    viewer._release_if_idle()
    assert viewer._GRAPH is not None and not docs.closed
    # idle past the threshold -> released
    viewer._last_access = viewer.time.monotonic() - (viewer.IDLE_RELEASE_SECONDS + 1)
    viewer._release_if_idle()
    assert viewer._GRAPH is None and viewer._DOCS is None and docs.closed


class _FakeGR:
    def exists(self):
        return True

    def entities(self):
        return [{"name": "Dominique", "type": "PERSON",
                 "description": "a person", "text_unit_ids": ["tu1"]}]

    def entity(self, name):
        return self.entities()[0] if name == "Dominique" else None

    def relationships(self):
        return [{"source": "Dominique", "target": "QIS",
                 "description": "works at", "weight": 1, "text_unit_ids": ["tu1"]}]

    def relationships_of(self, name):
        return self.relationships()

    def community_reports(self):
        return [{"community": "0", "level": 0, "title": "Work",
                 "summary": "s", "full_content": "f"}]

    def community(self, cid):
        return self.community_reports()[0] if cid == "0" else None

    def provenance_docs(self, ids):
        return ["notes:/w.md"]


def test_home_shows_graphrag_card(seeded, monkeypatch):
    client, _, _, _ = seeded
    monkeypatch.setattr(viewer, "_gr", lambda: _FakeGR())
    body = _text(client.get("/"))
    assert "GraphRAG" in body and "community reports" in body
    assert "/graphrag/entity" in body and "/graphrag/community" in body


def test_graphrag_entity_page_with_provenance(seeded, monkeypatch):
    client, _, _, _ = seeded
    monkeypatch.setattr(viewer, "_gr", lambda: _FakeGR())
    body = _text(client.get("/graphrag/entity", query_string={"id": "Dominique"}))
    assert "graphrag entity" in body
    assert "a person" in body and "QIS" in body
    # cross-DB: provenance links to our Document Store doc
    assert 'id=notes:/w.md' in body or "notes:/w.md" in body


def test_graphrag_community_page(seeded, monkeypatch):
    client, _, _, _ = seeded
    monkeypatch.setattr(viewer, "_gr", lambda: _FakeGR())
    body = _text(client.get("/graphrag/community", query_string={"id": "0"}))
    assert "community report" in body and "Work" in body


def test_parser_defaults():
    args = viewer.build_parser().parse_args([])
    assert args.host == "127.0.0.1" and args.port == 5000 and args.reload is False
