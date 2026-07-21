# Module 4 — Chat (simple mode)

`rag chat` answers questions over the embedded corpus. Only **simple mode**
(*retrieve-then-read*) is implemented: each user turn retrieves the most relevant
chunks, fuses them across embedding models, and lets a chat LLM answer from that
context. The **agentic** mode (the LLM plans its own retrieval, reads raw
documents, searches the web) is still in [`../PLAN.md`](../PLAN.md).

```sh
rag chat                                  # default chat model, all embedding indices
rag chat --model qwen2.5-7b --top-k 12    # explicit model, more context
rag chat --models qwen3-8b,bm25           # restrict which indices are queried
rag chat -v                               # also print what was retrieved
```

## Flow

For each question:

1. **Retrieve** from every selected embedding model (`rag/chat/retrieve.py`):
   - **dense** models embed the query (with the model's `query_instruction`) and do
     a vector search;
   - **sparse** (BM25) models turn the query into a local term-frequency vector and
     let Qdrant apply IDF at query time.
   A model with no collection yet simply contributes nothing.
2. **Fuse** the per-model ranked lists with **Reciprocal Rank Fusion** (RRF,
   `k = 60`): a chunk's score is `Σ 1/(k + rank)` over the lists it appears in, so a
   dense hit and a lexical hit for the same chunk reinforce each other. This is the
   hybrid-search step (semantic recall + exact-token matches).
3. **Reconstruct text.** Vector payloads store only chunk *identity*
   (`source_type`, `doc_id`, `set_id`, `part_id`, `chunk`), never the text. So the
   winning chunks are rebuilt deterministically: fetch the part's translated
   Markdown blob and re-chunk it with the **same** `chunk_size`/`chunk_overlap` the
   embed pass used (`embedding.chunk_size` / `chunk_overlap`), then take the named
   chunk index. A chunk whose part/index no longer exists (content changed since
   embedding) is skipped.
4. **Answer.** The chunks are formatted into a numbered context block and appended
   to the current user turn; the LLM is told to answer only from the context and
   cite passages by number. The answer is **streamed** token-by-token (important on
   a CPU-only box).

Conversation history keeps only the plain question/answer turns — the retrieved
context is attached to the *current* turn and not stored — so the context window
does not balloon over a long session.

### Cited sources

After each answer, a **`Sources:`** block lists the passages the model actually
cited (`[n]` markers parsed from the reply; hallucinated/out-of-range numbers are
dropped). Each line is `[n] <type> — <title>`, rendered as an **OSC 8 terminal
hyperlink** to the *original* document — the extractor's `source_url` (e.g. the
email plugin's mail-link), the same link the db-browser uses, not a db-browser
URL. When stdout is not an interactive terminal (piped, `NO_COLOR`, `TERM=dumb`),
the URL is shown in plain text instead; a document whose plugin offers no link
shows the label alone.

## Chat models

Configured under `chat` in the YAML, using the **same two backends** as embedding
models: native **Ollama** (`/api/chat`) or an **OpenAI-compatible** endpoint
(`/chat/completions`). Both are streamed with the standard library (no extra
dependency), mirroring `rag/embed/backends.py`. An OpenAI-compatible API key, if
needed, is resolved through the keyring (`api_key_name`), never from the YAML.

```yaml
chat:
  default_model: qwen2.5-7b     # used when --model is omitted
  retrieval:
    top_k: 8                    # chunks fed to the model per turn (override: --top-k)
  models:
    - id: qwen2.5-7b
      backend: ollama           # or: openai
      host: http://localhost:11434   # ollama; use api_base for an openai endpoint
      model: qwen2.5:7b-instruct
      num_ctx: 8192             # ollama context window (tokens)
      temperature: 0.2
      timeout: 600              # generous — CPU generation is slow
      # api_key_name: <secret>  # openai backend only; resolved via keyring
```

**Local CPU default.** On a CPU-only box with 32 GB RAM, a 7B instruct model (e.g.
`qwen2.5:7b-instruct`, ~5 GB at Q4, multilingual for DE/EN mail) runs comfortably
and reuses the Ollama server the embedder already uses. Pull it once with
`ollama pull qwen2.5:7b-instruct`. Swap to a larger model once better hardware is
available.

## REPL

The loop is [readline](https://docs.python.org/3/library/readline.html)-backed, so
it has line editing and a persistent history file at `storage/chat_history`. Type a
question at the `>` prompt; `/exit` (or `/quit`, `/q`, or **Ctrl-D**) leaves.
**Ctrl-C** at the prompt clears the line; **Ctrl-C** during an answer aborts just
that answer and returns to the prompt. Readline is optional — chat still works
without it (plain input, no history) on platforms that lack the module.

## Concurrency

Chat is a **reader**: it reads the metadata DB (WAL) freely while the pipeline
writes. The vector store follows its configured mode — a Qdrant **server** handles
concurrent access, so chat and `embed` can run together; the **embedded-local**
store is opened for the session and (per the single-writer policy in
[operations.md](operations.md#concurrency-model)) cannot overlap an `embed` run.
