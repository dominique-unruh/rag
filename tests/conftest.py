"""Shared fixtures. No network: mocks stand in for the LLM/embeddings."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from llama_index.core import MockEmbedding, Settings
from llama_index.core.llms import MockLLM


@pytest.fixture(autouse=True)
def _mock_models(monkeypatch):
    """Every test runs with mock LLM/embeddings so nothing calls out.

    Reading `Settings.llm` when unset lazily builds a real OpenAI client (which
    demands an API key), so we install mocks directly instead of saving/restoring
    the previous value, and set a dummy key as a belt-and-braces guard.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)
    yield


def _write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path):
    """A temp project dir with a valid config.yaml and a notes dir.

    Secrets are no longer a separate file: the api key is inlined as "none" and
    email is disabled by default. Returns (dir, write_config) so tests can
    rewrite config.yaml with overrides.
    """
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\nAlice met Bob.\n", encoding="utf-8")

    def write_config(**over) -> None:
        base = {
            "api_base": "http://127.0.0.1:9/v1",
            "chat_model": "m",
            "embed_model": "e",
            "embed_dim": 8,
            "api_key_line": "  api_key: none",
            "notes_dir": str(notes),
            "qdrant_path": str(tmp_path / "storage/qdrant"),
            "storage_dir": str(tmp_path / "storage"),
        }
        base.update(over)
        _write(
            tmp_path / "config.yaml",
            f"""
            models:
              api_base: {base['api_base']}
              chat_model: {base['chat_model']}
              embed_model: {base['embed_model']}
              embed_dim: {base['embed_dim']}
            {base['api_key_line']}
            email:
              enabled: false
            notes:
              enabled: true
              directory: {base['notes_dir']}
            qdrant:
              path: {base['qdrant_path']}
              collection: personal_rag
            graph:
              llm_extraction: false
              max_triplets_per_chunk: 8
            storage:
              dir: {base['storage_dir']}
            chunking:
              chunk_size: 1024
              chunk_overlap: 64
            """,
        )

    write_config()
    return tmp_path, write_config


@pytest.fixture
def secret_file(tmp_path: Path):
    """Factory: write a secret to a 0600 file, return its path (str)."""

    def make(name: str, value: str, mode: int = 0o600) -> str:
        path = tmp_path / name
        path.write_text(value, encoding="utf-8")
        os.chmod(path, mode)
        return str(path)

    return make
