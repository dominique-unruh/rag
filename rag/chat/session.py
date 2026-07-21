"""Simple-mode chat session (PLAN §7).

A line-based REPL (readline-backed, so it has history + editing) that, for each
question, retrieves relevant chunks, asks the chat LLM to answer from them, and
streams the answer. This is *simple mode* / *retrieve-then-read*; the agentic mode
where the LLM drives its own retrieval is still in ``PLAN.md``.

Concurrency: chat is a **reader**. The metadata DB (WAL) is read freely alongside
the pipeline; the vector store follows its configured mode (a Qdrant server allows
concurrent access, the embedded-local store is opened for the session — see
``doc/operations.md``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ..blobs import BlobStore
from ..config import Config
from ..db import Database
from ..embed.models import ModelSpec, chunk_params, load_models
from ..embed.vectorstore import open_vector_store
from ..errors import ConfigError, RagError
from ..secrets import Secrets
from .llm import Message, stream_chat
from .models import ChatModelSpec, get_chat_model, retrieval_top_k
from .prompt import build_user_message, system_prompt
from .retrieve import RetrievedChunk, Retriever
from .terminal import cited_indices, hyperlink, supports_hyperlinks

# Filenames/subdirs under the storage directory.
_DB_FILENAME: str = "metadata.db"
_BLOBS_SUBDIR: str = "blobs"
# Readline history file (per storage dir) so line history survives sessions.
_HISTORY_FILENAME: str = "chat_history"

# In-REPL commands (typed at the prompt) that are handled locally, not sent to the LLM.
_EXIT_COMMANDS: frozenset[str] = frozenset({"/exit", "/quit", "/q"})


def _setup_readline(history_path: Path) -> object | None:
    """Enable readline line-editing + persistent history; return the module or ``None``.

    ``readline`` is stdlib but absent on some platforms; chat still works without
    it (plain :func:`input`), just without history/editing.
    """
    try:
        import readline
    except ImportError:  # pragma: no cover - platform without readline
        return None
    try:
        readline.read_history_file(str(history_path))
    except (FileNotFoundError, OSError):
        # No prior history (first run) or an unreadable file — start fresh.
        pass
    return readline


def _save_history(readline_mod: object | None, history_path: Path) -> None:
    """Persist the readline history to ``history_path`` (best effort)."""
    if readline_mod is None:
        return
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline_mod.write_history_file(str(history_path))  # type: ignore[attr-defined]
    except OSError:  # pragma: no cover - history is a nicety, never fatal
        pass


def _select_embedding_specs(config: Config, embed_model_ids: list[str] | None) -> list[ModelSpec]:
    """Return the embedding :class:`ModelSpec`s to retrieve from.

    Defaults to every configured embedding model (so a dense + sparse pair gives
    hybrid retrieval); ``embed_model_ids`` restricts to a chosen subset.
    """
    registry = load_models(config.raw)
    if not registry:
        raise ConfigError(
            "no embedding models configured; chat needs at least one to retrieve from"
        )
    if embed_model_ids is None:
        return list(registry.values())
    specs: list[ModelSpec] = []
    for model_id in embed_model_ids:
        if model_id not in registry:
            available = ", ".join(sorted(registry))
            raise ConfigError(f"unknown embedding model {model_id!r}; available: {available}")
        specs.append(registry[model_id])
    return specs


def _answer_turn(
    spec: ChatModelSpec,
    history: list[Message],
    query: str,
    retriever: Retriever,
    embed_specs: list[ModelSpec],
    top_k: int,
    secrets: Secrets,
    *,
    verbose: bool,
) -> str:
    """Retrieve for ``query``, stream an answer, and return the full answer text.

    Prints retrieval progress (when ``verbose``) and the streamed answer to stdout.
    History holds only the plain Q&A turns; the retrieved context is attached to the
    *current* user turn only (not stored), keeping the context window from ballooning.
    """
    chunks = retriever.retrieve(query, embed_specs, top_k)
    # Send: system + prior plain turns + this turn's context-augmented question.
    augmented = build_user_message(query, chunks)
    messages: list[Message] = [
        {"role": "system", "content": system_prompt()},
        *history,
        {"role": "user", "content": augmented},
    ]
    if verbose:
        # Summarise what was retrieved (grounding) and the request about to be sent.
        print(f"[fused to {len(chunks)} passage(s) for context]", file=sys.stderr)
        for index, chunk in enumerate(chunks, start=1):
            label = chunk.title or chunk.doc_id
            print(f"  [{index}] {chunk.source_type}:{chunk.doc_id} — {label}", file=sys.stderr)
        # context size drives the model's latency, so report it before sending.
        context_chars = sum(len(m["content"]) for m in messages)
        print(
            f"[querying chat model {spec.id!r}: {len(messages)} message(s), "
            f"~{context_chars} chars; streaming reply...]",
            file=sys.stderr,
        )
    # Stream the answer, echoing deltas as they arrive and accumulating the whole.
    started = time.monotonic()
    pieces: list[str] = []
    for delta in stream_chat(spec, messages, secrets=secrets):
        pieces.append(delta)
        print(delta, end="", flush=True)
    print()  # newline after the streamed answer
    answer = "".join(pieces)
    if verbose:
        # The reply was streamed above; report totals now that it is complete.
        elapsed = time.monotonic() - started
        print(
            f"[reply complete: {len(answer)} chars in {elapsed:.1f}s]", file=sys.stderr
        )
    # List the sources the answer actually cited, each linked to the original doc.
    _print_cited_sources(answer, chunks)
    return answer


def _print_cited_sources(answer: str, chunks: list[RetrievedChunk]) -> None:
    """Print a ``Sources:`` block for every ``[n]`` the answer cited.

    One line per cited passage — ``[n] <type> — <title>`` — where the label is an
    OSC 8 hyperlink to the original document (from the extractor's ``source_url``)
    when stdout is a terminal, else the URL is shown in plain text. Passages the
    model did not cite are omitted; nothing is printed if it cited none.
    """
    indices = cited_indices(answer, len(chunks))
    if not indices:
        return
    links_on = supports_hyperlinks(sys.stdout)
    print("\nSources:")
    for number in indices:
        chunk = chunks[number - 1]  # citations are 1-based into the context list
        # Label: document type + title (falling back to the doc id when untitled).
        label = f"{chunk.source_type} — {chunk.title or chunk.doc_id}"
        if chunk.url:
            print(f"  [{number}] {hyperlink(chunk.url, label, enabled=links_on)}")
        else:
            # No source link available for this document — show the label alone.
            print(f"  [{number}] {label}")


def run_chat_simple(
    config: Config,
    *,
    model_id: str | None = None,
    top_k: int | None = None,
    embed_model_ids: list[str] | None = None,
    verbose: bool = False,
    secrets: Secrets | None = None,
) -> None:
    """Run the interactive simple-mode chat REPL until EOF/exit.

    Resolves the chat model (``model_id`` or ``chat.default_model``) and the
    embedding models to retrieve from, then loops: read a question, retrieve, and
    stream a grounded answer. ``top_k`` overrides ``chat.retrieval.top_k``.
    """
    resolver = secrets if secrets is not None else Secrets()
    spec = get_chat_model(config.raw, model_id)
    embed_specs = _select_embedding_specs(config, embed_model_ids)
    effective_top_k = top_k if top_k is not None else retrieval_top_k(config.raw)
    chunk_size, overlap = chunk_params(config.raw)

    storage = config.storage_dir
    history_path = storage / _HISTORY_FILENAME
    # Open the vector store once for the session (a server handles concurrency; the
    # embedded-local store is held for the session — see doc/operations.md).
    vectors = open_vector_store(config, secrets=resolver)
    database = Database.open(storage / _DB_FILENAME)
    blobs = BlobStore(storage / _BLOBS_SUBDIR)
    retriever = Retriever(
        vectors, database.conn, blobs,
        chunk_size=chunk_size, overlap=overlap, secrets=resolver, verbose=verbose,
    )
    readline_mod = _setup_readline(history_path)

    # Conversation history: plain Q&A turns only (context is re-attached per turn).
    history: list[Message] = []
    models_note = ", ".join(s.id for s in embed_specs)
    print(
        f"chat: model {spec.id!r}, retrieving top {effective_top_k} from [{models_note}].\n"
        f"Ask a question; /exit or Ctrl-D to quit.",
        file=sys.stderr,
    )
    try:
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                # Ctrl-D at the prompt: clean exit.
                print(file=sys.stderr)
                break
            except KeyboardInterrupt:
                # Ctrl-C at the prompt: abandon the current line, keep the session.
                print("^C", file=sys.stderr)
                continue
            if not query:
                continue
            if query in _EXIT_COMMANDS:
                break
            try:
                answer = _answer_turn(
                    spec, history, query, retriever, embed_specs,
                    effective_top_k, resolver, verbose=verbose,
                )
            except KeyboardInterrupt:
                # Ctrl-C mid-answer: stop this answer, drop the partial turn, continue.
                print("\n[interrupted]", file=sys.stderr)
                continue
            except RagError as exc:
                # A backend/config failure shouldn't kill the whole session.
                print(f"error: {exc}", file=sys.stderr)
                continue
            # Record only the plain turn so follow-ups have continuity without
            # re-sending every prior context block.
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
    finally:
        _save_history(readline_mod, history_path)
        vectors.close()
        database.close()
