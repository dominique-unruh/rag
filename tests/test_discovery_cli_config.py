"""Tests for plugin discovery, config loading and CLI option parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.cli import _parse_kv_options
from rag.config import Config
from rag.errors import ConfigError, PluginError
from rag.extract.base import Extractor
from rag.extract.discovery import discover_plugins, load_plugin


def test_discovery_finds_builtin_plugins() -> None:
    """The built-in plugin(s) are discovered, keyed by their directory name."""
    plugins = discover_plugins()
    assert "email" in plugins
    # Each mapping value is a concrete Extractor subclass.
    assert all(issubclass(cls, Extractor) for cls in plugins.values())


def test_load_unknown_plugin_raises() -> None:
    """Loading a non-existent source type raises a clear error."""
    with pytest.raises(PluginError):
        load_plugin("does-not-exist")


def test_config_storage_dir_and_plugin_subtree(tmp_path: Path) -> None:
    """Config resolves the storage dir and hands out plugin subtrees."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "storage:\n  dir: ./data\n"
        "extractors:\n  email:\n    host: imap.example.org\n"
    )
    config = Config.load(cfg_path)
    # Relative storage dir is resolved against the config file's directory.
    assert config.storage_dir == (tmp_path / "data").resolve()
    assert config.plugin_config("email") == {"host": "imap.example.org"}
    # A plugin with no configured section gets an empty mapping.
    assert config.plugin_config("missing") == {}


def test_config_missing_file_raises(tmp_path: Path) -> None:
    """Loading a missing config file raises :class:`ConfigError`."""
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "nope.yaml")


def test_parse_kv_options() -> None:
    """``key=value`` tokens parse into a dict; malformed tokens raise."""
    assert _parse_kv_options(["count=100", "folder=INBOX"]) == {
        "count": "100",
        "folder": "INBOX",
    }
    with pytest.raises(SystemExit):
        _parse_kv_options(["bad"])
