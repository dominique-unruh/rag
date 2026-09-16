# RAG Pipeline — Design Plan (remaining work)

A local, single-user RAG system that ingests personal data (email, notes, …),
turns it into clean Markdown, embeds it into a vector index, and exposes a chat
interface over it.

The pipeline has four stages — **Extract → Translate → Embed → Chat**. The first
three are **implemented**, as is chat **simple mode**; their architecture is
documented under [`doc/`](doc/) (start at [`doc/README.md`](doc/README.md)). This
file tracks only what is **not yet built**.

---

## Module 4 — Chat

Chat interface over the RAG, with selectable modes and LLM models.

- **Simple mode** (*retrieve-then-read*) — **implemented** (`rag chat`, see
  [`doc/chat.md`](doc/chat.md)). For each turn: retrieve the most relevant chunks
  for the user's query, fuse them, and let the LLM answer from that context. A
  readline-backed REPL (line editing + persistent history under
  `storage/chat_history`).
- **Complex / agentic mode** — **not built**. The LLM can actively access the RAG,
  the raw documents, the translations, and perform web searches; it plans its own
  retrieval strategy.

Model and mode are selectable on the command line. Chat LLMs use the same backend
options as embedding models (see [`doc/embedding.md`](doc/embedding.md)): an
**OpenAI-compatible** endpoint (hosted, or local via Ollama's `/v1` / LM Studio /
vLLM / llama.cpp) or **native Ollama**.

**Retrieval / fusion (implemented).** When several embedding models are populated,
retrieval queries each selected model's index and **fuses** the ranked lists via
Reciprocal Rank Fusion into one result set — this is where a dense model and a
sparse/lexical model combine into hybrid search (semantic recall + exact token
matches). Which models participate is a query-time choice (`--models`); a model
with incomplete coverage simply contributes what it has. (A cross-encoder reranker
could later sit after fusion as an optional query-time stage — no index change.)

**Still to build for module 4:** agentic mode; optional answer streaming niceties
(a spinner/cancellation polish); a cross-encoder reranker after fusion.

---

## More extractors (not built)

New source plugins to add once email has proven the design (each a subdirectory
under `extract/plugins/`, `source_type` = directory name):

- **Zulip** — messages/streams from a Zulip account.
- **Other chat accounts** — further chat platforms (Signal, Messenger, WhatsApp, Telegram, …).
- **Notes (in Obsidian)**.
- **RWTH Intranet** — in particular regulations, *Rundschreiben*, and similar
  official documents.

---

## Translation — media transcription translator (not built; deferred until GPU)

Audio/video → text as a **translator** (module 2), a sibling of `pdf`/`docx`/`xlsx`
— **not** a new extractor (the `documents` plugin already ingests media files as
blobs with `audio/*`/`video/*` mime). **Deferred until a GPU is available**: local
Whisper + a vision model are too slow on the current CPU-only box. See the
GPU-arrival reminder in auto-memory.

**Scope.** Two layers, merged into one timestamped Markdown transcript:
1. **Audio → text** — Whisper (via `faster-whisper`), multilingual (DE/EN).
2. **Visual → text** — for video, sample scene-change keyframes and describe them
   with a vision-LLM; plus OCR (tesseract) for screencasts/slides (on-screen text).
   Essential for speechless videos, where the visual track *is* the content.

**Reuse existing OSS — do not hand-roll** frame sampling / scene-dedup / whisper:
- `HUANGCHIHHUNGLeo/claude-real-video` (`crv`, MIT, library-importable) — scene-aware
  deduplicated keyframes + real timestamps + local `faster-whisper` transcript. It
  does *not* describe visuals itself. **Preferred building block.**
- `byjlw/video-analyzer` (Apache-2.0) — turnkey: keyframes → Llama 3.2 Vision +
  Whisper → combined description (`analysis.json`). GPU-oriented; least code but
  least control. Fallback if we want turnkey.
- General recipe everywhere: ffmpeg frames → open VLM (Qwen2.5-VL / MiniCPM-V /
  InternVL / llava / llama3.2-vision) + Whisper. (`pyVideoTrans` = subtitles/dubbing,
  not visual description — not relevant.)

**Recommended shape.** `translators/transcribe.py` (`Transcribe`): `crv` for
keyframes+transcript → run deduped frames through **Ollama vision** → merge to
`[hh:mm:ss] (speech|visual|on-screen) …` Markdown. Dispatch routes
`mime.startswith("audio/")`/`"video/"` **before** the text fallback. Store detected
language + duration in part metadata.

**Blocker to resolve first — translator config plumbing.** Translators today are
stateless module-level instances that receive **no config** (`translate(document)`
only; `pdf.py` etc. take nothing). Whisper/vision need config (model id, device,
`compute_type`, language hint, frame cap). Cleanest: reuse the central `models:`
registry (as `embed`/`chat` do) and have the pipeline pass the selected translator
its config subtree (e.g. an optional `configure(config)` called after
`load_translator`). This change is a prerequisite for this translator.

**Guardrails.** Duration/size cap (skip huge files with an error, like the
`documents` byte cap); scene-dedup + max-frames cap bounds vision cost; deps
(`faster-whisper`, `crv`, vision via Ollama, optional `tesseract`) are hard —
fail loud if missing (per project convention).

---

## Translation — image description + OCR translator (not built; deferred until GPU)

Image (`image/png`, `image/jpeg`, `image/gif`, `image/webp`, `image/tiff`, …) →
Markdown as a **translator** (module 2), a sibling of `pdf`/`docx`/`xlsx` — **not**
a new extractor (the `documents` plugin already ingests images as blobs with
`image/*` mime, and the `archive` translator already dispatches images pulled out
of archives). Shares the **same GPU blocker and the same config-plumbing blocker**
as the media-transcription translator above.

**Scope.** Turn one image into indexable Markdown by combining two signals:
1. **Visual description** — a vision-LLM (VLM) caption / scene understanding, so a
   photo, diagram or screenshot with no embedded text still yields searchable prose
   ("a whiteboard sketch of the extract→translate→embed pipeline with three boxes …").
2. **OCR (on-screen text)** — the literal text in the image (scanned page,
   screenshot, slide, photographed sign). Essential for screenshots/scans where the
   text *is* the content.

Merge both into one part: a `## Description` section from the VLM followed by a
`## Text (OCR)` section, plus detected-language / dimensions in part metadata.

**Key finding — the two layers can collapse into one VLM call.** Modern VLMs
(Qwen3-VL, Qwen2.5-VL, MiniCPM-V) do *both* description **and** high-quality
multilingual OCR in a single pass — so the primary path is one VLM prompt that
returns "describe + transcribe". Standalone OCR is the **CPU-only fallback** (and a
cheaper always-on layer) when no VLM is available. This mirrors how the video plan
folds visual + on-screen text together, but here it is one model, not two tools.

**Reuse existing OSS — do not hand-roll:**
- **Vision via Ollama** (same transport as `embed`/`chat`): `qwen3-vl` /
  `qwen2.5-vl:7b` (Jan 2025, strong multilingual OCR) or `minicpm-v` (best
  character-level OCR for handwriting / odd fonts). **Preferred building block** —
  it does description + OCR at once and reuses the existing OpenAI-compatible /
  native-Ollama backend code. Local ⇒ needs a GPU (12 GB fits a 7–8B model at 4-bit).
- **`Docling` (IBM, MIT)** — turnkey document-image pipeline: layout analysis +
  TableFormer + EasyOCR, outputs structured Markdown, has LlamaIndex/LangChain RAG
  hooks. Fallback turnkey option if we want layout/table fidelity (the
  `video-analyzer` equivalent for images); heavier, GPU-oriented.
- **CPU-only OCR fallback** — `tesseract` (via `pytesseract`, multilingual DE/EN) or
  `PaddleOCR` (PP-OCR, better on complex layouts / non-Latin / curved text). Runs
  today with no GPU; this is the layer that could ship *before* the GPU arrives.
- Image decoding via `Pillow` (normalise/resize/convert before OCR or before base64
  to the VLM).

**GPU verdict (what the user asked to assess).** The **OCR half is CPU-only** and
could ship now (tesseract/PaddleOCR). The **VLM-description half needs a GPU** to run
locally, exactly like the video translator's vision layer — *or* a hosted
OpenAI-compatible vision endpoint, which needs no local GPU but sends personal
images to a third party (privacy) and costs per call. Because the user wants
**both** description and OCR merged, the whole translator is **deferred until a GPU
is available**, tracked with the media-transcription translator. (If we later want
value sooner, ship the tesseract/PaddleOCR OCR-only layer first and add the VLM
description layer when the GPU lands.)

**Blocker to resolve first — translator config plumbing** (shared with the video
translator; see that section). A VLM needs config (model id, endpoint, language
hint, max image dimension); OCR needs a language hint and tesseract/Paddle path.
Same fix: an optional `configure(config)` after `load_translator`, reusing the
central `models:` registry.

**Recommended shape.** `translators/image.py` (`Image`): decode with Pillow →
if a VLM is configured, one "describe + transcribe" call; else CPU OCR only → emit
`## Description` / `## Text (OCR)` Markdown. Dispatch routes
`mime.startswith("image/")` **before** the text fallback (add an `_IMAGE_TYPES`
route or a `startswith` check next to the `audio/`·`video/` one).

**Guardrails.** Pixel/byte-size cap (downscale huge images before the VLM to bound
cost/tokens, skip absurd ones with an error like the `documents` byte cap); deps
(`Pillow`, `pytesseract`/`paddleocr`, vision via Ollama) are hard — fail loud if
missing (per project convention).

---

## Maintenance — remaining tasks (not built)

The `maintenance` dispatcher exists with `db-browser`, `delete` and `stats` (see
[`doc/operations.md`](doc/operations.md)). Still to build:

- **GC-like tasks** (filter-driven): remove unreferenced blobs, prune inactive
  translations, drop orphaned embeddings — each under the write lock, aborting on
  uncertainty, with a dry-run/report mode (the safety principle is documented in
  [`doc/operations.md`](doc/operations.md)).

---

## Open risks

- **Vector-store concurrency [open].** The embedded-Qdrant policy (embed takes an
  exclusive lock for its whole run; chat releases on idle) leaves chat retrieval
  unavailable during embedding and needs retry/backoff plus an idle-timer
  lifecycle. Residual risks: lock races, a stale lock if `embed` crashes mid-run,
  and up-to-1s chat stalls. Fallback if this bites: `sqlite-vec` (real WAL
  reader/writer concurrency, no custom locking). Full rationale in
  [`doc/design-decisions.md`](doc/design-decisions.md) (item 1).

---

## Remaining build order

1. **Module 4 (chat)**: simple mode is built; **agentic mode** next. Fusion of
   multiple embedding indices is in place.
2. **Additional extraction plugins** (see *More extractors* above), now that email
   has proven the design.
3. **Maintenance**: GC.
4. Iterate: more filters, context-aware translation, more source types.
