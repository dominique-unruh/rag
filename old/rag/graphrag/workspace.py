"""Generate a Microsoft GraphRAG workspace (settings.yaml) from our Config.

GraphRAG is configured by a settings.yaml in a root dir; it writes its output
parquet under <root>/output. We point its models at the same OpenAI-compatible
endpoint the rest of rag uses. Prompt keys are omitted so GraphRAG uses its
built-in defaults (their config defaults are None -> built-in prompt).
"""

from __future__ import annotations

from pathlib import Path

from .. import stores
from ..config import Config

CHAT_ID = "default_completion_model"
EMBED_ID = "default_embedding_model"


def _settings_yaml(cfg: Config) -> str:
    return f"""\
completion_models:
  {CHAT_ID}:
    type: litellm
    model_provider: openai
    model: {cfg.models.chat_model}
    api_base: {cfg.models.api_base}
    api_key: "{cfg.models.api_key}"

embedding_models:
  {EMBED_ID}:
    type: litellm
    model_provider: openai
    model: {cfg.models.embed_model}
    api_base: {cfg.models.api_base}
    api_key: "{cfg.models.api_key}"

input:
  type: text
  base_dir: input

chunking:
  size: {cfg.chunking.chunk_size}
  overlap: {cfg.chunking.chunk_overlap}

input_storage:
  type: file
  base_dir: input
output_storage:
  type: file
  base_dir: output
reporting:
  type: file
  base_dir: logs
cache:
  type: json
  storage:
    type: file
    base_dir: cache

vector_store:
  type: lancedb
  db_uri: output/lancedb
  vector_size: {cfg.models.embed_dim}   # must match the embedding model's width
                                         # (default 3072 != our model -> lancedb crash)

embed_text:
  embedding_model_id: {EMBED_ID}
extract_graph:
  completion_model_id: {CHAT_ID}
summarize_descriptions:
  completion_model_id: {CHAT_ID}
community_reports:
  completion_model_id: {CHAT_ID}
"""


def ensure_workspace(cfg: Config) -> Path:
    """Create the GraphRAG root + settings.yaml (idempotent). Returns the root."""
    root = stores.graphrag_dir(cfg)
    (root / "input").mkdir(exist_ok=True)
    settings = root / "settings.yaml"
    settings.write_text(_settings_yaml(cfg), encoding="utf-8")
    return root


def output_dir(cfg: Config) -> Path:
    return stores.graphrag_dir(cfg) / "output"
