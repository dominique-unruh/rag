"""Unit tests for module 4 (chat, simple mode).

Cover the pure, network-free pieces: the chat-model registry, the LLM body
builders / stream-delta parsers, RRF fusion, chunk-text reconstruction, and prompt
construction. The interactive REPL and live backends are not exercised here.
"""

from __future__ import annotations

import pytest

from rag.chat.llm import (
    _ollama_body,
    _ollama_delta,
    _openai_body,
    _openai_delta,
)
from rag.chat.models import ChatModelSpec, get_chat_model, load_chat_models, retrieval_top_k
from rag.chat.prompt import build_user_message, format_context, system_prompt
from rag.chat.retrieve import RetrievedChunk, _chunk_at, rrf_fuse
from rag.chat.terminal import cited_indices, hyperlink
from rag.embed.chunk import chunk_text
from rag.errors import ConfigError


# --- chat model registry -------------------------------------------------


def _chat_config(**overrides: object) -> dict:
    """Return a minimal config raw mapping with a ``chat`` section."""
    chat = {
        "default_model": "local",
        "models": [
            {"id": "local", "backend": "ollama", "host": "http://x:11434", "model": "m"}
        ],
    }
    chat.update(overrides)
    return {"chat": chat}


def test_load_chat_models_parses_entry() -> None:
    """A configured chat model is parsed into a ChatModelSpec with its knobs."""
    raw = {"chat": {"models": [{
        "id": "q", "backend": "ollama", "host": "http://h:1", "model": "qwen",
        "num_ctx": 4096, "temperature": 0.3, "max_tokens": 256,
    }]}}
    models = load_chat_models(raw)
    assert set(models) == {"q"}
    spec = models["q"]
    assert (spec.backend, spec.model, spec.num_ctx, spec.max_tokens) == ("ollama", "qwen", 4096, 256)
    assert spec.temperature == 0.3


def test_get_chat_model_uses_default() -> None:
    """With no id given, chat.default_model selects the model."""
    spec = get_chat_model(_chat_config(), None)
    assert spec.id == "local"


def test_get_chat_model_single_model_no_default() -> None:
    """A lone configured model is used even without an explicit default."""
    raw = {"chat": {"models": [{"id": "solo", "backend": "ollama", "host": "h", "model": "m"}]}}
    assert get_chat_model(raw, None).id == "solo"


def test_get_chat_model_ambiguous_without_default() -> None:
    """Several models and no default => an actionable ConfigError."""
    raw = {"chat": {"models": [
        {"id": "a", "backend": "ollama", "host": "h", "model": "m"},
        {"id": "b", "backend": "ollama", "host": "h", "model": "m"},
    ]}}
    with pytest.raises(ConfigError):
        get_chat_model(raw, None)


def test_get_chat_model_unknown_id() -> None:
    """An unknown --model id is rejected."""
    with pytest.raises(ConfigError):
        get_chat_model(_chat_config(), "nope")


def test_get_chat_model_none_configured() -> None:
    """No chat section at all is a clear ConfigError."""
    with pytest.raises(ConfigError):
        get_chat_model({}, None)


def test_bad_backend_rejected() -> None:
    """An unsupported backend fails loudly."""
    with pytest.raises(ConfigError):
        load_chat_models({"chat": {"models": [{"id": "x", "backend": "bogus", "model": "m"}]}})


def test_retrieval_top_k_default_and_override() -> None:
    """top_k falls back to the default and honours the configured value."""
    assert retrieval_top_k({}) == 8
    assert retrieval_top_k({"chat": {"retrieval": {"top_k": 3}}}) == 3
    with pytest.raises(ConfigError):
        retrieval_top_k({"chat": {"retrieval": {"top_k": 0}}})


# --- LLM body builders / delta parsers -----------------------------------


def _spec(**kw: object) -> ChatModelSpec:
    """Build a ChatModelSpec with test defaults."""
    base = dict(id="i", backend="ollama", model="m", host="http://h:1")
    base.update(kw)
    return ChatModelSpec(**base)  # type: ignore[arg-type]


def test_ollama_body_nests_options_and_streams() -> None:
    """Ollama options are nested and only set knobs are sent; stream is on."""
    body = _ollama_body(_spec(num_ctx=2048, temperature=0.1), [{"role": "user", "content": "hi"}])
    assert body["stream"] is True
    assert body["options"] == {"num_ctx": 2048, "temperature": 0.1}


def test_ollama_body_omits_options_when_unset() -> None:
    """With no knobs set, no options block is sent (model defaults apply)."""
    body = _ollama_body(_spec(), [{"role": "user", "content": "hi"}])
    assert "options" not in body


def test_openai_body_top_level_knobs() -> None:
    """OpenAI knobs are top-level (max_tokens/temperature), not nested."""
    spec = _spec(backend="openai", api_base="http://h/v1", temperature=0.5, max_tokens=100)
    body = _openai_body(spec, [{"role": "user", "content": "hi"}])
    assert body["stream"] is True
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 100


def test_delta_parsers_extract_content() -> None:
    """Delta parsers pull text out of a chunk and tolerate empty/odd shapes."""
    assert _ollama_delta({"message": {"content": "ab"}}) == "ab"
    assert _ollama_delta({"done": True}) == ""
    assert _openai_delta({"choices": [{"delta": {"content": "xy"}}]}) == "xy"
    assert _openai_delta({"choices": [{"delta": {}}]}) == ""
    assert _openai_delta({}) == ""


# --- RRF fusion ----------------------------------------------------------


def test_rrf_fuse_rewards_agreement() -> None:
    """A chunk both lists rank highly beats one ranked highly by only one list."""
    a = ("email", "1", 10, 0)
    b = ("email", "2", 20, 0)
    c = ("email", "3", 30, 0)
    dense = [a, b, c]   # a is dense's top
    sparse = [a, c, b]  # a is sparse's top too
    fused = rrf_fuse([dense, sparse])
    keys = [k for k, _ in fused]
    assert keys[0] == a  # agreed-upon top wins
    # a's score is strictly greater than any single-list contribution.
    scores = dict(fused)
    assert scores[a] > scores[b] and scores[a] > scores[c]


def test_rrf_fuse_empty() -> None:
    """Fusing nothing yields nothing."""
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# --- chunk reconstruction ------------------------------------------------


def test_chunk_at_matches_embed_chunking() -> None:
    """_chunk_at returns exactly the chunk the embed pass would have embedded."""
    text = "abcdefghij" * 5  # 50 chars
    size, overlap = 20, 5
    expected = chunk_text(text, size, overlap)
    for i, exp in enumerate(expected):
        assert _chunk_at(text, size, overlap, i) == exp


def test_chunk_at_out_of_range() -> None:
    """An out-of-range index (stale content) returns None, not an error."""
    assert _chunk_at("short", 20, 5, 7) is None
    assert _chunk_at("", 20, 5, 0) is None


# --- prompt construction -------------------------------------------------


def _chunk(n: int, text: str, title: str | None = "T", url: str | None = None) -> RetrievedChunk:
    """Build a RetrievedChunk for prompt tests."""
    return RetrievedChunk(
        source_type="email", doc_id=str(n), part_id=n, chunk=0,
        text=text, title=title, date="2026-01-01", url=url, score=1.0,
    )


def test_format_context_numbers_passages() -> None:
    """Context passages are numbered [1], [2] with a source label and the text."""
    out = format_context([_chunk(1, "alpha"), _chunk(2, "beta")])
    assert "[1] (email — T — 2026-01-01)" in out
    assert "alpha" in out and "[2]" in out and "beta" in out


def test_format_context_empty() -> None:
    """No chunks yields an explicit 'no passages' marker (never an empty prompt)."""
    assert "no relevant passages" in format_context([])


def test_build_user_message_includes_question_and_context() -> None:
    """The augmented user turn carries both the context and the raw question."""
    msg = build_user_message("what is X?", [_chunk(1, "alpha")])
    assert "alpha" in msg and "what is X?" in msg
    assert system_prompt()  # non-empty system instruction


# --- terminal citations / hyperlinks -------------------------------------


def test_cited_indices_parses_and_bounds() -> None:
    """Citation parsing handles [n] and [n, m], dedups, and drops out-of-range."""
    answer = "See [1] and [2, 3]; also [1] again and a bogus [9]."
    assert cited_indices(answer, count=3) == [1, 2, 3]  # 9 dropped, order preserved


def test_cited_indices_none() -> None:
    """An answer with no citations yields no indices."""
    assert cited_indices("no citations here", count=5) == []


def test_hyperlink_osc8_when_enabled() -> None:
    """With links enabled, the text is wrapped in an OSC 8 sequence hiding the URL."""
    out = hyperlink("https://e/x", "email — Subj", enabled=True)
    assert out.startswith("\033]8;;https://e/x\a")
    assert "email — Subj" in out
    assert out.endswith("\033]8;;\a")


def test_hyperlink_plain_fallback() -> None:
    """With links disabled (piped/opted out), the URL is shown in plain text."""
    assert hyperlink("https://e/x", "email — Subj", enabled=False) == "email — Subj (https://e/x)"
