"""Load and validate config.yaml.

One committable file. Credentials are NOT inlined: config.yaml points at
external files via `models.api_key_file` and `email.password_file`, each holding
a single secret (the api key / imap password). The loader reads config.yaml,
resolves those files to their contents, validates required keys, and expands
user paths. Any missing file or required key fails fast with a clear,
actionable message.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from .errors import UserError

CONFIG_FILE = "config.yaml"


class ConfigError(UserError):
    """Raised for any missing file, missing key, or invalid config value."""


def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _read_secret_file(raw_path: str, label: str) -> str:
    """Read a secret from an external file; strip surrounding whitespace.

    Fails fast if missing/empty; warns (does not fail) on group/other access.
    """
    path = Path(_expand(raw_path))
    if not path.exists():
        raise ConfigError(f"{label} points to a missing file: {path}")
    if not path.is_file():
        raise ConfigError(f"{label} must point to a file, not a directory: {path}")
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"warning: {path} is group/other-accessible; run: chmod 600 {path}")
    secret = path.read_text(encoding="utf-8").strip()
    if not secret:
        raise ConfigError(f"{label} file is empty: {path}")
    return secret


def _load_yaml(path: Path, kind: str, example: str) -> dict:
    if not path.exists():
        raise ConfigError(
            f"{kind} not found at {path}. Copy {example} to {path.name} and edit it."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a YAML mapping, got {type(data).__name__}.")
    return data


def _require(cfg: dict, path: str) -> Any:
    """Fetch a dotted key or raise ConfigError naming the missing key."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node or node[part] is None:
            raise ConfigError(f"Missing required config key: {path}")
        node = node[part]
    return node


@dataclass
class ModelsConfig:
    api_base: str
    chat_model: str
    embed_model: str
    embed_dim: int
    api_key: str
    temperature: float
    # Only set for models that support Matryoshka/reduced output dims. When None,
    # no `dimensions` param is sent and the model returns its native width.
    embed_dimensions: int | None = None


@dataclass
class EmailConfig:
    enabled: bool
    host: str
    username: str
    password: str
    folder: str
    since: date  # resolved server-side SINCE cutoff


@dataclass
class NotesConfig:
    enabled: bool
    directory: str


@dataclass
class QdrantConfig:
    collection: str
    path: str | None = None
    url: str | None = None


@dataclass
class GraphConfig:
    llm_extraction: bool
    max_triplets_per_chunk: int


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass
class Config:
    models: ModelsConfig
    email: EmailConfig
    notes: NotesConfig
    qdrant: QdrantConfig
    graph: GraphConfig
    chunking: ChunkingConfig
    storage_dir: str
    raw: dict = field(repr=False, default_factory=dict)


def _resolve_since(email: dict) -> date:
    """since_date (fixed) overrides since_days (relative)."""
    if email.get("since_date") is not None:
        val = email["since_date"]
        if isinstance(val, date) and not isinstance(val, datetime):
            return val
        if isinstance(val, datetime):
            return val.date()
        try:
            return datetime.strptime(str(val), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ConfigError(
                f"email.since_date must be YYYY-MM-DD, got {val!r}"
            ) from exc
    days = email.get("since_days")
    if days is None:
        raise ConfigError("email requires one of: since_date or since_days")
    try:
        days = int(days)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"email.since_days must be an integer, got {days!r}") from exc
    return date.today() - timedelta(days=days)


KEYRING_SERVICE = "ragproto"


def _keyring():
    """Return a usable keyring module, or None if none is available.

    keyring auto-selects a backend, but a misconfigured default (e.g. the
    libsecret backend on a box without PyGObject) *raises* during detection —
    which would otherwise take down every get/set. If the default is broken we
    fall back to the pure-Python SecretService backend (talks to
    gnome-keyring/KWallet over DBus; `secretstorage` ships as a dependency).
    """
    try:
        import keyring
    except Exception:
        return None
    try:
        keyring.get_keyring()  # forces backend detection
        return keyring
    except Exception:
        pass  # configured backend is broken; try SecretService explicitly
    try:
        from keyring.backends import SecretService

        backend = SecretService.Keyring()
        _ = backend.priority  # raises if secretstorage/DBus is unavailable
    except Exception:
        return None
    keyring.set_keyring(backend)
    return keyring


def _keyring_get_or_prompt(account: str, prompt_fn, label: str) -> str:
    """Look up `account` in the keyring; on miss, prompt and store it back.

    The lookup and store are best-effort: a broken/locked keyring degrades to a
    plain prompt rather than failing the run.
    """
    kr = _keyring()
    if kr is not None:
        try:
            stored = kr.get_password(KEYRING_SERVICE, account)
        except Exception as exc:  # backend present but broken/locked
            print(f"warning: keyring lookup failed ({exc}); will prompt.")
            stored = None
        if stored:
            return stored

    secret = prompt_fn()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, account, secret)
        except Exception as exc:
            print(f"warning: could not store {label} in keyring: {exc}")
    return secret


def _prompt_secret(prompt: str, missing_hint: str) -> str:
    import getpass
    import sys

    if not sys.stdin.isatty():
        raise ConfigError(missing_hint)
    return getpass.getpass(prompt)


def _prompt_password(username: str) -> str:
    return _prompt_secret(
        f"IMAP password for {username}: ",
        f"IMAP password for {username} not found in config or keyring, and no "
        f"terminal is available to prompt. Set email.password / email.password_file, "
        f"or pre-seed the keyring (service '{KEYRING_SERVICE}', user '{username}').",
    )


def _prompt_api_key(api_base: str) -> str:
    return _prompt_secret(
        f"API key for {api_base}: ",
        f"API key for {api_base} not found in config or keyring, and no terminal is "
        f"available to prompt. Set models.api_key / models.api_key_file, or pre-seed "
        f"the keyring (service '{KEYRING_SERVICE}', user 'api_key@{api_base}').",
    )


def _resolve_api_key(models_raw: dict, api_base: str) -> str:
    """Resolve the API key, in order:

    1. inline `models.api_key` (use "none" for local servers that ignore it),
    2. `models.api_key_file` (external file),
    3. desktop keyring (service 'ragproto', user = 'api_key@<api_base>'),
    4. interactive prompt — the entered value is stored back into the keyring.
    """
    if models_raw.get("api_key") is not None:
        return str(models_raw["api_key"])
    key_file = models_raw.get("api_key_file")
    if key_file:
        return _read_secret_file(str(key_file), "models.api_key_file")
    return _keyring_get_or_prompt(
        f"api_key@{api_base}", lambda: _prompt_api_key(api_base), "api key"
    )


def _resolve_password(email_raw: dict, username: str) -> str:
    """Resolve the IMAP password, in order:

    1. `email.password` inline in config,
    2. `email.password_file` (external file),
    3. desktop keyring (service 'ragproto', user = imap username),
    4. interactive prompt — the entered value is then stored back into the keyring
       so subsequent runs are non-interactive.
    """
    if email_raw.get("password"):
        return str(email_raw["password"])
    pw_file = email_raw.get("password_file")
    if pw_file:
        return _read_secret_file(str(pw_file), "email.password_file")
    return _keyring_get_or_prompt(
        username, lambda: _prompt_password(username), "password"
    )


def load(config_dir: str | os.PathLike = ".") -> Config:
    """Load and validate config from config_dir; return a typed Config."""
    root = Path(config_dir)
    cfg = _load_yaml(root / CONFIG_FILE, "config.yaml", "config.yaml.example")

    models_raw = cfg.get("models", {}) or {}
    api_base = str(_require(cfg, "models.api_base"))
    models = ModelsConfig(
        api_base=api_base,
        chat_model=str(_require(cfg, "models.chat_model")),
        embed_model=str(_require(cfg, "models.embed_model")),
        embed_dim=int(_require(cfg, "models.embed_dim")),
        api_key=_resolve_api_key(models_raw, api_base),
        temperature=float(models_raw.get("temperature", 0.1)),
        embed_dimensions=(
            int(models_raw["embed_dimensions"])
            if models_raw.get("embed_dimensions") is not None
            else None
        ),
    )

    email_raw = cfg.get("email", {}) or {}
    email_enabled = bool(email_raw.get("enabled", False))
    if email_enabled:
        username = str(_require(cfg, "email.username"))
        email = EmailConfig(
            enabled=True,
            host=str(_require(cfg, "email.host")),
            username=username,
            password=_resolve_password(email_raw, username),
            folder=str(email_raw.get("folder", "INBOX")),
            since=_resolve_since(email_raw),
        )
    else:
        email = EmailConfig(
            enabled=False,
            host="",
            username="",
            password="",
            folder="INBOX",
            since=date.today(),
        )

    notes_raw = cfg.get("notes", {}) or {}
    notes_enabled = bool(notes_raw.get("enabled", False))
    if notes_enabled:
        notes = NotesConfig(
            enabled=True,
            directory=_expand(str(_require(cfg, "notes.directory"))),
        )
    else:
        notes = NotesConfig(enabled=False, directory="")

    if not (email.enabled or notes.enabled):
        raise ConfigError("At least one source must be enabled: email or notes.")

    q_raw = cfg.get("qdrant", {}) or {}
    q_path = q_raw.get("path")
    q_url = q_raw.get("url")
    if bool(q_path) == bool(q_url):
        raise ConfigError("qdrant requires exactly one of: path or url (not both/neither).")
    qdrant = QdrantConfig(
        collection=str(_require(cfg, "qdrant.collection")),
        path=_expand(str(q_path)) if q_path else None,
        url=str(q_url) if q_url else None,
    )

    graph_raw = cfg.get("graph", {}) or {}
    graph = GraphConfig(
        llm_extraction=bool(graph_raw.get("llm_extraction", True)),
        max_triplets_per_chunk=int(graph_raw.get("max_triplets_per_chunk", 8)),
    )

    chunk_raw = cfg.get("chunking", {}) or {}
    chunking = ChunkingConfig(
        chunk_size=int(chunk_raw.get("chunk_size", 1024)),
        chunk_overlap=int(chunk_raw.get("chunk_overlap", 64)),
    )

    storage_dir = _expand(str(_require(cfg, "storage.dir")))

    return Config(
        models=models,
        email=email,
        notes=notes,
        qdrant=qdrant,
        graph=graph,
        chunking=chunking,
        storage_dir=storage_dir,
        raw=cfg,
    )
