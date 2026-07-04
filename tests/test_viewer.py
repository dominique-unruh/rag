"""viewer: Flask routes render items and cross-link references (no network)."""

from __future__ import annotations

import json
import os

import pytest
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

from rag import config, stores, viewer


@pytest.fixture
def seeded(project):
    """Persist a small graph: (Dominique)-[WORKS_AT]->(QIS), sourced from a doc."""
    root, _ = project
    cfg = config.load(root)

    # A markdown file under the notes directory so the obsidian translator applies.
    doc = os.path.join(cfg.notes.directory, "meetings", "w.md")
    node_content = json.dumps({"id_": "c1", "metadata": {"file_name": "w.md"},
                               "text": "hi"})
    chunk = ChunkNode(
        text="# Meeting\nDominique works at QIS in Berlin.\n",
        properties={"file_path": doc, "file_name": "w.md",
                    "_node_content": node_content},
    )
    cid = chunk.id
    dom = EntityNode(name="Dominique", label="entity",
                     properties={"triplet_source_id": cid, "file_path": doc})
    qis = EntityNode(name="QIS", label="entity",
                     properties={"triplet_source_id": cid})

    gs = SimplePropertyGraphStore()
    gs.upsert_nodes([chunk, dom, qis])
    gs.upsert_relations([
        Relation(label="WORKS_AT", source_id="Dominique", target_id="QIS",
                 properties={"triplet_source_id": cid}),
        Relation(label="SOURCE", source_id=cid, target_id=doc,
                 properties={"file_path": doc}),
    ])
    stores.persist_graph_store(gs, cfg)

    viewer._CFG = cfg
    return viewer.app.test_client(), cid, doc


def _text(resp):
    return resp.get_data(as_text=True)


def test_index_lists_entities_and_documents(seeded):
    client, _, doc = seeded
    r = client.get("/")
    assert r.status_code == 200
    body = _text(r)
    assert "Dominique" in body
    assert doc in body
    assert "2 entities" in body and "documents" in body


def test_entity_shows_relations_and_linked_target(seeded):
    client, cid, _ = seeded
    r = client.get("/item", query_string={"kind": "entity", "id": "Dominique"})
    body = _text(r)
    assert "WORKS_AT" in body
    assert 'id=QIS' in body  # target is a clickable link
    # triplet_source_id property links to the chunk
    assert f'id={cid}' in body or "triplet_source_id" in body


def test_relation_page_links_both_endpoints(seeded):
    client, _, _ = seeded
    r = client.get("/relation", query_string={
        "source": "Dominique", "label": "WORKS_AT", "target": "QIS"})
    body = _text(r)
    assert "Dominique" in body and "QIS" in body
    assert "id=Dominique" in body and "id=QIS" in body


def test_chunk_shows_text_and_extracted_entities(seeded):
    client, cid, _ = seeded
    r = client.get("/item", query_string={"kind": "chunk", "id": cid})
    body = _text(r)
    assert "Dominique works at QIS" in body
    assert "Dominique" in body and "QIS" in body  # entities extracted here


def test_document_shows_metadata_chunks_entities(seeded):
    client, cid, doc = seeded
    r = client.get("/item", query_string={"kind": "document", "id": doc})
    body = _text(r)
    assert "file_name" in body  # metadata table
    assert "Dominique" in body  # entity from this doc
    assert f"id={cid}"[:8] in body  # chunk listed


def test_missing_item_404(seeded):
    client, _, _ = seeded
    r = client.get("/item", query_string={"kind": "entity", "id": "Nobody"})
    assert r.status_code == 404


def test_history_panel_present(seeded):
    client, _, _ = seeded
    body = _text(client.get("/"))
    assert 'id="history"' in body or "id=history" in body
    assert "rag_view_history" in body  # localStorage-backed history


def test_search_form_present_on_pages(seeded):
    client, _, _ = seeded
    assert "action=/search" in _text(client.get("/"))


def test_search_text_ranks_entities(seeded):
    client, _, _ = seeded
    body = _text(client.get("/search", query_string={"q": "Domin"}))
    assert "Graph matches" in body
    assert "Dominique" in body and "id=Dominique" in body


def test_search_relation_and_chunk_matches(seeded):
    client, cid, _ = seeded
    body = _text(client.get("/search", query_string={"q": "WORKS"}))
    assert "WORKS_AT" in body  # relation hit
    body2 = _text(client.get("/search", query_string={"q": "Berlin"}))
    assert "chunk" in body2.lower()  # chunk text match


def test_search_exact_id_jumps_to_item(seeded):
    client, _, _ = seeded
    body = _text(client.get("/search", query_string={"q": "Dominique"}))
    # rendered as the entity item page, not a results list
    assert "Relations (" in body and "Properties" in body


def test_search_shortened_id_prefix_jumps(seeded):
    client, cid, _ = seeded
    body = _text(client.get("/search", query_string={"q": cid[:10]}))
    assert "chunk" in body.lower()
    assert "Text" in body  # chunk item page has a Text section


def test_search_exact_document_path(seeded):
    client, _, doc = seeded
    body = _text(client.get("/search", query_string={"q": doc}))
    assert "Metadata" in body and "Chunks (" in body


def test_search_semantic_unavailable_is_graceful(seeded):
    # config points at a dead endpoint -> embedding fails -> section degrades.
    client, _, _ = seeded
    body = _text(client.get("/search", query_string={"q": "anything", "semantic": "1"}))
    assert "Semantic matches" in body
    assert "unavailable" in body


def test_search_semantic_qdrant_lock_is_graceful(seeded, monkeypatch):
    # Embedded Qdrant lock (another instance holds it) must not 500 the page.
    client, _, _ = seeded

    # Keep the fixture's MockEmbedding (no-op configure so it isn't replaced),
    # so embedding succeeds and we reach the Qdrant open, which fails on the lock.
    monkeypatch.setattr(viewer.runtime, "configure", lambda cfg: None)

    def boom(cfg):
        raise RuntimeError("Storage folder ... is already accessed by another instance")

    monkeypatch.setattr(viewer.stores, "open_qdrant_client", boom)

    resp = client.get("/search", query_string={"q": "x", "semantic": "1"})
    assert resp.status_code == 200  # not a 500
    body = _text(resp)
    assert "Qdrant unavailable" in body
    assert "already accessed by another instance" in body


def test_entity_type_header_is_just_kind(seeded):
    client, _, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "entity", "id": "Dominique"}))
    assert 'class="kind entity"' in body
    assert "object ·" not in body  # dropped the "object · " prefix


def test_kind_badges_color_coded(seeded):
    client, cid, doc = seeded
    assert 'class="kind chunk"' in _text(
        client.get("/item", query_string={"kind": "chunk", "id": cid}))
    assert 'class="kind document"' in _text(
        client.get("/item", query_string={"kind": "document", "id": doc}))
    assert 'class="kind relation"' in _text(client.get(
        "/relation", query_string={"source": "Dominique", "label": "WORKS_AT", "target": "QIS"}))


def test_history_has_letter_badges(seeded):
    client, _, _ = seeded
    body = _text(client.get("/"))
    # history JS derives a kind letter and styles it like the type badge
    assert "LETTER" in body and "relation:'R'" in body
    assert "'hb '+kind" in body


def test_file_path_obsidian_link(seeded):
    client, _, doc = seeded
    body = _text(client.get("/item", query_string={"kind": "entity", "id": "Dominique"}))
    # notes file -> obsidian://open with vault + url-encoded, extension-stripped file
    assert "obsidian://open?vault=notes&amp;file=meetings%2Fw" in body
    assert ".md" not in body.split("obsidian://open")[1][:60]  # extension dropped
    dbody = _text(client.get("/item", query_string={"kind": "document", "id": doc}))
    assert "obsidian://open?vault=notes&amp;file=meetings%2Fw" in dbody


def test_path_link_unrecognized_is_plain(seeded):
    # _CFG is set by the fixture; paths outside the notes vault are not linkified.
    _client, _, _ = seeded
    assert "<a" not in viewer._path_link("/etc/passwd")
    # a /home/... markdown file NOT under the notes dir is not an obsidian file
    other = viewer._path_link("/home/unruh/Documents/other/note.md")
    assert "<a" not in other and "obsidian" not in other


def test_obsidian_uses_fixed_notes_vault(seeded):
    _client, _, doc = seeded  # doc is under the notes dir
    url = viewer._translate_path(doc)
    assert url.startswith("obsidian://open?vault=notes&file=")


def test_messageid_to_mid_link(seeded):
    _client, _, _ = seeded
    mid = "CAOCutbvzFcsK7ScRQTFMcze_XTLAp8cVMy6b+HqkmYNggbCR6A@mail.gmail.com"
    url = viewer._translate_path(mid)
    assert url.startswith("mid:")
    assert "%2B" in url and "+" not in url  # '+' correctly quoted
    assert url.endswith("@mail.gmail.com")  # '@' kept readable
    link = viewer._path_link(mid)
    assert 'href="mid:' in link


def test_messageid_translator_ignores_paths_and_plain(seeded):
    _client, _, _ = seeded
    assert viewer._messageid_translator(None, "/home/x/n.md") is None  # path
    assert viewer._messageid_translator(None, "Dominique") is None     # no '@'


def test_document_page_email_id_is_mid_link(project):
    root, _ = project
    cfg = config.load(root)
    mid = "abc+def@mail.example.org"
    chunk = ChunkNode(text="hi", properties={})
    gs = SimplePropertyGraphStore()
    gs.upsert_nodes([chunk])
    gs.upsert_relations([Relation(label="SOURCE", source_id=chunk.id, target_id=mid)])
    stores.persist_graph_store(gs, cfg)
    viewer._CFG = cfg

    body = viewer.app.test_client().get(
        "/item", query_string={"kind": "document", "id": mid}).get_data(as_text=True)
    assert 'href="mid:abc%2Bdef@mail.example.org"' in body


def test_path_translation_is_extensible(seeded):
    # Registry-based: adding a translator changes rendering; first match wins.
    _client, _, _ = seeded
    viewer._PATH_TRANSLATORS.insert(0, lambda cfg, p: "x://hit" if p == "/tmp/z" else None)
    try:
        assert 'href="x://hit"' in viewer._path_link("/tmp/z")
    finally:
        viewer._PATH_TRANSLATORS.pop(0)


def test_chunk_text_rendered_as_markdown(seeded):
    client, cid, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "chunk", "id": cid}))
    assert "class=md" in body
    assert "<h1>Meeting</h1>" in body  # '# Meeting' rendered, not shown literally


def test_render_md_handles_syntax():
    html_out = viewer._render_md("# H\n\n**b** and `code`\n\n- a\n- b")
    assert "<h1>H</h1>" in html_out
    assert "<strong>b</strong>" in html_out and "<code>code</code>" in html_out


def test_node_content_rendered_as_json_tree(seeded):
    client, cid, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "chunk", "id": cid}))
    assert "class=jt" in body  # foldable JSON viewer used
    assert "<span class=jkey>metadata</span>" in body  # nested key rendered
    assert "<details class=jt open>" in body  # top-level expanded


def test_json_tree_scalars_and_nesting():
    out = viewer._json_tree({"a": 1, "b": [True, None, "s"], "c": {}})
    assert "<span class=jkey>a</span>" in out
    assert "<span class=jnum>1</span>" in out
    assert "<span class=jbool>true</span>" in out and "<span class=jnull>null</span>" in out
    assert "<span class=jstr>&quot;s&quot;</span>" in out or '"s"' in out
    assert "<details class=jt" in out


def test_try_json_fallback_on_invalid():
    assert viewer._try_json("not json") is None
    assert viewer._try_json("123") is None       # scalar, not object/array
    assert viewer._try_json('{"k": 1}') == {"k": 1}
    assert viewer._try_json("{bad json}") is None  # no error, just None


def test_copy_button_on_properties(seeded):
    client, _, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "entity", "id": "Dominique"}))
    assert "<button class=copy" in body
    assert "data-copy=" in body
    assert "_COPY_JS" not in body  # the constant name must not leak
    assert "navigator.clipboard" in body  # the handler is inlined


def test_semantic_checkbox_has_tooltip(seeded):
    client, _, _ = seeded
    body = _text(client.get("/"))
    assert "vector similarity search" in body  # tooltip text present
    assert "class=info" in body  # ⓘ hint marker


def test_property_descriptions_loaded():
    assert viewer._prop_desc("entity", "file_type")  # from common section
    assert viewer._prop_desc("chunk", "_node_content")  # kind-specific
    assert viewer._prop_desc("entity", "no_such_property") is None


def test_prop_th_tooltip_and_icon():
    th = viewer._prop_th("chunk", "_node_content")
    assert "title=" in th and "class=pinfo" in th and "ⓘ" in th
    plain = viewer._prop_th("entity", "no_such_property")
    assert "title=" not in plain and "pinfo" not in plain


def test_item_page_shows_property_tooltip(seeded):
    client, _, _ = seeded
    body = _text(client.get("/item", query_string={"kind": "entity", "id": "Dominique"}))
    assert "class=pinfo" in body  # circled-i marker rendered
    assert "Absolute path to the source file" in body  # tooltip text present


def test_parser_defaults():
    args = viewer.build_parser().parse_args([])
    assert args.host == "127.0.0.1" and args.port == 5000
