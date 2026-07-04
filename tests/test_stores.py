"""Storage wiring: paths, extractor gating, store roundtrips, qdrant client."""

from __future__ import annotations

from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core import Document

from rag import config, stores


def test_paths_under_storage_dir(project):
    root, _ = project
    cfg = config.load(root)
    assert stores.docstore_path(cfg).name == "docstore.json"
    assert stores.graph_store_path(cfg).name == "property_graph_store.json"
    # _storage_dir side effect: dir gets created
    assert stores.docstore_path(cfg).parent.is_dir()


def test_extractors_gate_off(project):
    root, _ = project
    cfg = config.load(root)  # llm_extraction: false in fixture
    ex = stores.build_extractors(cfg)
    assert len(ex) == 1 and isinstance(ex[0], ImplicitPathExtractor)


def test_extractors_gate_on(project):
    root, _ = project
    cfg = config.load(root)
    cfg.graph.llm_extraction = True
    ex = stores.build_extractors(cfg)  # uses Settings.llm (MockLLM from fixture)
    assert len(ex) == 2
    assert any(isinstance(e, SimpleLLMPathExtractor) for e in ex)
    assert any(isinstance(e, ImplicitPathExtractor) for e in ex)


def test_docstore_roundtrip(project):
    root, _ = project
    cfg = config.load(root)
    ds = SimpleDocumentStore()
    ds.add_documents([Document(text="hi", id_="x")])
    stores.persist_docstore(ds, cfg)

    loaded = stores.load_docstore(cfg)
    assert "x" in loaded.docs


def test_docstore_load_missing_is_fresh(project):
    root, _ = project
    cfg = config.load(root)
    assert stores.load_docstore(cfg).docs == {}


def test_graph_store_roundtrip(project):
    root, _ = project
    cfg = config.load(root)
    gs = SimplePropertyGraphStore()
    gs.upsert_nodes([EntityNode(label="entity", name="Alice")])
    stores.persist_graph_store(gs, cfg)
    assert stores.graph_store_path(cfg).exists()

    loaded = stores.load_graph_store(cfg)
    assert isinstance(loaded, SimplePropertyGraphStore)


def test_qdrant_embedded_client_and_vector_store(project):
    root, _ = project
    cfg = config.load(root)
    client = stores.open_qdrant_client(cfg)
    try:
        vs = stores.build_vector_store(client, cfg)
        assert vs.collection_name == "personal_rag"
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
