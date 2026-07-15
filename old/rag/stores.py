"""Shared storage wiring for all pipelines and the query side.

Local, embedded backends:
  - documents   -> SQLite Document Store   (documents.db)   [canonical corpus]
  - vectors     -> Qdrant                  (qdrant/)
  - graph       -> Kùzu                     (graph.kuzu/)

Extraction writes documents.db; the embed and graph pipelines read it and write
Qdrant / Kùzu; chat + viewer read Qdrant + Kùzu.
"""

from __future__ import annotations

import os
from pathlib import Path

import kuzu
import qdrant_client
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.schema import TransformComponent
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from .config import Config
from .documents import DocumentStore

DOCUMENTS_DB = "documents.db"
GRAPH_DB = "graph.kuzu"


def _storage_dir(cfg: Config) -> Path:
    d = Path(cfg.storage_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def documents_db_path(cfg: Config) -> Path:
    return _storage_dir(cfg) / DOCUMENTS_DB


def graph_db_path(cfg: Config) -> Path:
    return _storage_dir(cfg) / GRAPH_DB


def logs_dir(cfg: Config) -> Path:
    d = _storage_dir(cfg) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def graphrag_dir(cfg: Config) -> Path:
    """Root workspace for the Microsoft GraphRAG index (separate from Kùzu)."""
    d = _storage_dir(cfg) / "graphrag"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Document Store -------------------------------------------------------

def open_document_store(cfg: Config, check_same_thread: bool = True) -> DocumentStore:
    return DocumentStore(documents_db_path(cfg), check_same_thread=check_same_thread)


# --- Qdrant ---------------------------------------------------------------

def open_qdrant_client(cfg: Config) -> qdrant_client.QdrantClient:
    """Embedded (local `path`) or remote (`url`). Exactly one is set (validated)."""
    if cfg.qdrant.path:
        os.makedirs(cfg.qdrant.path, exist_ok=True)
        return qdrant_client.QdrantClient(path=cfg.qdrant.path)
    return qdrant_client.QdrantClient(url=cfg.qdrant.url)


def build_vector_store(
    client: qdrant_client.QdrantClient, cfg: Config
) -> QdrantVectorStore:
    return QdrantVectorStore(collection_name=cfg.qdrant.collection, client=client)


# --- Kùzu graph -----------------------------------------------------------

def open_graph_store(cfg: Config) -> tuple[KuzuPropertyGraphStore, kuzu.Database]:
    """Open the embedded Kùzu graph DB. Caller closes the returned db.

    Kùzu is single-writer per directory (and the store writes its schema on init,
    so read-only opening is not possible). Open it in one process at a time.
    """
    db = kuzu.Database(str(graph_db_path(cfg)))
    return KuzuPropertyGraphStore(db), db


def close_graph_db(db: kuzu.Database) -> None:
    close = getattr(db, "close", None)
    if callable(close):
        close()


# --- Graph extractors + read index ----------------------------------------

def build_extractors(cfg: Config) -> list[TransformComponent]:
    """ImplicitPathExtractor always on (cheap, structural). LLM extractor is the
    cost lever, gated by graph.llm_extraction."""
    extractors: list[TransformComponent] = [ImplicitPathExtractor()]
    if cfg.graph.llm_extraction:
        extractors.append(
            DynamicLLMPathExtractor(
                llm=Settings.llm,
                max_triplets_per_chunk=cfg.graph.max_triplets_per_chunk,
            )
        )
    return extractors


def open_read_index(
    cfg: Config,
    graph_store: KuzuPropertyGraphStore,
    vector_store: QdrantVectorStore,
) -> PropertyGraphIndex:
    """Read-only PropertyGraphIndex for chat retrieval (Qdrant vectors + graph)."""
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        vector_store=vector_store,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        kg_extractors=[],
        embed_kg_nodes=False,
        # Embedded Qdrant has no async client; keep retrieval on the sync path.
        use_async=False,
    )
