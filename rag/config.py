"""Central YAML configuration (PLAN §10).

A single YAML file holds everything except secrets. Each extraction plugin gets
its own subtree, keyed by ``source_type`` under the top-level ``plugins`` map;
the framework hands a plugin just that subtree so plugins never see unrelated
keys. Secrets are resolved separately (see :mod:`rag.secrets`).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

# Default location of the storage directory if the config omits ``storage.dir``.
# Everything the pipeline persists (DB, blobs, plugin state) lives beneath it.
_DEFAULT_STORAGE_DIR: str = "./storage"


class Config:
    """Parsed view over the project's YAML configuration file."""

    def __init__(self, data: Mapping[str, Any], source_path: Path) -> None:
        """Wrap already-parsed config ``data`` loaded from ``source_path``.

        ``source_path`` is retained so relative paths in the config can be
        resolved against the config file's own directory, and for error messages.
        """
        # Raw parsed YAML mapping (top-level keys: storage, plugins, models, …).
        self._data: Mapping[str, Any] = data
        # Directory containing the config file, used to resolve relative paths.
        self._base_dir: Path = source_path.resolve().parent
        # Path of the config file itself (diagnostics only).
        self.source_path: Path = source_path

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load and parse the YAML config at ``path``.

        Raises :class:`ConfigError` if the file is missing or does not parse to a
        mapping.
        """
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        try:
            # ``safe_load`` avoids executing arbitrary YAML tags.
            parsed = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"failed to parse config {path}: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ConfigError(f"config root must be a mapping, got {type(parsed).__name__}")
        return cls(parsed, path)

    def _resolve_path(self, raw: str) -> Path:
        """Expand ``~`` and resolve ``raw`` relative to the config file's directory.

        Absolute paths and ``~``-relative paths are honoured as-is; other
        relative paths are anchored at the config directory so behaviour does not
        depend on the process's working directory.
        """
        expanded = Path(raw).expanduser()
        if expanded.is_absolute():
            return expanded
        # Anchor project-relative paths at the config file location.
        return (self._base_dir / expanded).resolve()

    @property
    def storage_dir(self) -> Path:
        """Directory under which the DB, blobs and plugin state are stored."""
        # Read the nested ``storage.dir`` key, falling back to the default.
        storage = self._data.get("storage", {})
        raw = storage.get("dir", _DEFAULT_STORAGE_DIR) if isinstance(storage, Mapping) else _DEFAULT_STORAGE_DIR
        return self._resolve_path(str(raw))

    @property
    def vector_store(self) -> Mapping[str, Any]:
        """Return the top-level ``vector_store`` config subtree (empty if absent).

        Holds the Qdrant deployment settings: ``url`` (server mode; omit for the
        embedded-local store under ``storage/qdrant``), ``prefer_grpc``,
        ``timeout``, ``on_disk`` and an optional ``quantization`` subsection. The
        server API key is not a config key — it is resolved from the keyring under
        a fixed secret name (see :mod:`rag.embed.vectorstore`).
        """
        section = self._data.get("vector_store", {})
        if not isinstance(section, Mapping):
            raise ConfigError("top-level 'vector_store' must be a mapping")
        return section

    def plugin_config(self, source_type: str) -> Mapping[str, Any]:
        """Return the config subtree for the extractor ``source_type``.

        Looks up ``extractors.<source_type>``; returns an empty mapping when the
        extractor has no configured section (an extractor may need none).
        """
        extractors = self._data.get("extractors", {})
        if not isinstance(extractors, Mapping):
            raise ConfigError("top-level 'extractors' must be a mapping")
        section = extractors.get(source_type, {})
        if not isinstance(section, Mapping):
            raise ConfigError(f"extractors.{source_type} must be a mapping")
        return section

    @property
    def raw(self) -> Mapping[str, Any]:
        """The full parsed config mapping (escape hatch for uncommon keys)."""
        return self._data
