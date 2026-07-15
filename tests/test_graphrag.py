"""GraphRAG pipeline: config, input frame, parquet reader (no real index run)."""

from __future__ import annotations

import pandas as pd

from rag import config
from rag.documents import DocumentStore
from rag.graphrag import _frame, build_parser
from rag.graphrag.workspace import ensure_workspace
from rag.graphrag.reader import GraphRagStore


def test_parser_flags_match_graph():
    args = build_parser().parse_args(["--max", "5", "--docs", "notes:*", "-v", "--rebuild"])
    assert args.max == 5 and args.docs == "notes:*"
    assert args.verbose is True and args.rebuild is True
    assert build_parser().parse_args(["--communities"]).communities is True


def test_communities_requires_graph(project):
    import pytest
    from rag import stores
    from rag.errors import UserError
    from rag.graphrag import run_communities

    root, _ = project
    cfg = config.load(root)
    with pytest.raises(UserError, match="No graph index yet"):
        run_communities(cfg)  # no entities.parquet -> refuse


def test_communities_runs_community_workflows(project, monkeypatch):
    import types

    from rag import config as cfgmod
    from rag.graphrag import _COMMUNITY_WORKFLOWS, run_communities
    from rag.graphrag.workspace import output_dir

    root, _ = project
    cfg = cfgmod.load(root)
    od = output_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    (od / "entities.parquet").write_bytes(b"")  # graph exists

    seen = {}

    async def fake_build_index(gr_config, input_documents=None, is_update_run=False,
                               verbose=False):
        seen["workflows"] = gr_config.workflows
        seen["input"] = input_documents
        seen["update"] = is_update_run

        class R:
            error = None
            workflow = "w"

        return [R()]

    import graphrag.api
    import graphrag.config.load_config as lc
    monkeypatch.setattr(graphrag.api, "build_index", fake_build_index)
    monkeypatch.setattr(lc, "load_config", lambda root: types.SimpleNamespace(embed_text=types.SimpleNamespace()))

    run_communities(cfg)
    assert seen["workflows"] == _COMMUNITY_WORKFLOWS
    assert seen["input"] is None and seen["update"] is False  # no docs, not an update


def test_frame_builds_documents_df(tmp_path):
    store = DocumentStore(tmp_path / "d.db")
    store.upsert("notes", "/a.md", "alpha", {"file_name": "a.md"})
    store.upsert("email", "m@h", "gamma", {"subject": "hi"})

    df = _frame(store, ["notes:/a.md", "email:m@h"])
    assert list(df.columns) == ["id", "text", "title"]
    assert set(df["id"]) == {"notes:/a.md", "email:m@h"}
    assert df[df["id"] == "notes:/a.md"]["title"].iloc[0] == "a.md"   # source_label
    assert df[df["id"] == "email:m@h"]["title"].iloc[0] == "hi"       # subject


def test_ensure_workspace_writes_settings(project):
    root, _ = project
    cfg = config.load(root)
    gr_root = ensure_workspace(cfg)
    settings = (gr_root / "settings.yaml").read_text()
    assert cfg.models.api_base in settings
    assert cfg.models.chat_model in settings and cfg.models.embed_model in settings
    assert "completion_models" in settings and "embedding_models" in settings


def test_reader_reads_parquet(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    pd.DataFrame([
        {"title": "Dominique", "type": "PERSON",
         "description": "a person", "text_unit_ids": ["tu1"]},
    ]).to_parquet(out / "entities.parquet")
    pd.DataFrame([
        {"source": "Dominique", "target": "QIS",
         "description": "works at", "weight": 1.0, "text_unit_ids": ["tu1"]},
    ]).to_parquet(out / "relationships.parquet")
    pd.DataFrame([
        {"community": 0, "level": 0, "title": "Work",
         "summary": "sum", "full_content": "full"},
    ]).to_parquet(out / "community_reports.parquet")
    pd.DataFrame([
        {"id": "tu1", "document_ids": ["notes:/a.md"]},
    ]).to_parquet(out / "text_units.parquet")

    gr = GraphRagStore(out)
    assert gr.exists()
    ents = gr.entities()
    assert ents[0]["name"] == "Dominique" and ents[0]["type"] == "PERSON"
    assert gr.entity("Dominique")["description"] == "a person"
    assert gr.relationships_of("Dominique")[0]["target"] == "QIS"
    assert gr.community("0")["title"] == "Work"
    # provenance maps text_unit -> our doc id
    assert gr.provenance_docs(["tu1"]) == ["notes:/a.md"]


def test_sanitize_graphrag_embeddings(monkeypatch):
    import asyncio

    import graphrag.index.operations.embed_text.embed_text as et
    from rag.graphrag import _sanitize_graphrag_embeddings

    captured = {}

    async def fake_run_embed_text(texts, *args, **kwargs):
        captured["texts"] = texts
        return object()

    monkeypatch.setattr(et, "run_embed_text", fake_run_embed_text)
    monkeypatch.setattr(et, "_rag_sanitized", False, raising=False)

    _sanitize_graphrag_embeddings()  # wraps et.run_embed_text
    asyncio.run(et.run_embed_text(["a", "", "  \n", "b"], None, None, None, 0, 0, 0))
    assert captured["texts"] == ["a", "[empty]", "[empty]", "b"]  # empties replaced


def test_reader_missing_output_is_empty(tmp_path):
    gr = GraphRagStore(tmp_path / "nope")
    assert gr.exists() is False
    assert gr.entities() == [] and gr.community_reports() == []


def test_incremental_full_then_update(project, monkeypatch):
    from rag import stores
    from rag.graphrag import run_graphrag
    from rag.graphrag.workspace import output_dir

    root, _ = project
    cfg = config.load(root)
    s = stores.open_document_store(cfg)
    for i in range(6):  # >= MIN_FULL_BUILD_DOCS for the initial build
        s.upsert("notes", f"/d{i}.md", f"text {i}", {})
    s.close()

    calls = []

    async def fake_build_index(gr_config, input_documents=None, is_update_run=False,
                               verbose=False):
        calls.append({"ids": sorted(input_documents["id"]), "update": is_update_run,
                      "workflows": gr_config.workflows})
        od = output_dir(cfg)
        od.mkdir(parents=True, exist_ok=True)
        (od / "entities.parquet").write_bytes(b"")  # our pipeline's completion marker

        class R:
            error = None
            workflow = "w"

        return [R()]

    import types

    import graphrag.api
    import graphrag.config.load_config as lc
    monkeypatch.setattr(graphrag.api, "build_index", fake_build_index)
    monkeypatch.setattr(lc, "load_config", lambda root: types.SimpleNamespace(embed_text=types.SimpleNamespace()))

    from rag.graphrag import _FULL_WORKFLOWS, _UPDATE_WORKFLOWS

    run_graphrag(cfg)  # 1) full build over all docs, reduced (no community) pipeline
    assert calls[-1]["update"] is False and len(calls[-1]["ids"]) == 6
    assert calls[-1]["workflows"] == _FULL_WORKFLOWS
    assert "create_community_reports" not in calls[-1]["workflows"]

    run_graphrag(cfg)  # 2) nothing changed -> no index call
    assert len(calls) == 1

    s = stores.open_document_store(cfg)
    s.upsert("notes", "/new.md", "y", {})
    s.close()
    run_graphrag(cfg)  # 3) update run over just the new doc
    assert calls[-1]["ids"] == ["notes:/new.md"] and calls[-1]["update"] is True
    assert calls[-1]["workflows"] == _UPDATE_WORKFLOWS
