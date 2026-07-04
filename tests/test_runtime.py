"""Runtime wiring: model/embedding/splitter construction and Settings install."""

from __future__ import annotations

import pytest
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from rag import config, runtime


def test_build_llm(project):
    root, _ = project
    cfg = config.load(root)
    llm = runtime.build_llm(cfg)
    assert isinstance(llm, OpenAILike)
    assert llm.model == "m"
    assert llm.api_base == "http://127.0.0.1:9/v1"
    assert llm.is_chat_model is True
    assert llm.temperature == 0.1  # config default, low for grounding


def test_build_embedding_omits_dimensions_by_default(project):
    # No embed_dimensions in config -> don't send `dimensions` (native width).
    root, _ = project
    cfg = config.load(root)
    emb = runtime.build_embedding(cfg)
    assert isinstance(emb, OpenAILikeEmbedding)
    assert emb.dimensions is None


def test_build_embedding_sends_dimensions_when_opted_in(project):
    root, _ = project
    cfg = config.load(root)
    cfg.models.embed_dimensions = 256  # opt in to Matryoshka reduction
    emb = runtime.build_embedding(cfg)
    assert emb.dimensions == 256


def test_build_splitter(project):
    root, _ = project
    cfg = config.load(root)
    sp = runtime.build_splitter(cfg)
    assert isinstance(sp, SentenceSplitter)
    assert sp.chunk_size == 1024 and sp.chunk_overlap == 64


def test_configure_installs_settings(project):
    root, _ = project
    cfg = config.load(root)
    runtime.configure(cfg)
    assert isinstance(Settings.llm, OpenAILike)
    assert isinstance(Settings.embed_model, OpenAILikeEmbedding)
    assert isinstance(Settings.node_parser, SentenceSplitter)


def test_verify_embedding_dim_ok(project):
    # fixture installs MockEmbedding(embed_dim=8) and config embed_dim is 8.
    root, _ = project
    cfg = config.load(root)
    assert runtime.verify_embedding_dim(cfg) == 8


def test_verify_embedding_dim_mismatch(project):
    root, _ = project
    cfg = config.load(root)
    cfg.models.embed_dim = 999  # provider (mock) returns 8
    with pytest.raises(runtime.EmbeddingDimError, match="embed_dim=999.*returned 8"):
        runtime.verify_embedding_dim(cfg)
