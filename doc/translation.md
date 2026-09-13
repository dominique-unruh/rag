# Module 2 — Processing / Translation

Takes a document from `documents` and produces one or more Markdown translation
blobs, stored as a **translation set** (see [storage-model.md](storage-model.md))
in the DB + blob store.

**One source → several parts.** A single translation run may emit multiple output
blobs, each a *part* with its own metadata. Example: translating an email produces
a `body` part (the message text) plus one part per attachment
(`attachment:invoice.pdf`, an OCR'd image, …). The parts are grouped into one
translation set and share a single `is_active` flag, so they are always activated /
deactivated / reverted together — a document is never left with a half-active mix
of old and new parts. A simple document (e.g. a note) just produces a set with a
single `body` part.

**Possible processing steps** (composable, selected by filters):
- Data-format conversion (e.g. email → text).
- Image processing (OCR / picture recognition).
- AI rewriting: an LLM rewrites terse content into a more self-contained form for
  better embedding.
- **Context-aware rewriting**: the translator may query the *already-built RAG* to
  resolve context (e.g. what a first name refers to, overall topic).
- Access to the raw extraction result and its metadata.
- Identity translation: if the source is already plain text / Markdown, pass it
  through.

**Selection**: which documents to (re)translate and how is driven by arbitrary
filters — document type, source, age of last translation, etc.

**Output** (see the `translation_sets` / `translation_parts` schema in
[storage-model.md](storage-model.md)): a set row (source id, source_version,
run-level pipeline metadata, timestamp, git revision, `is_active`) plus one part
row per output blob (part_key, blob, per-part metadata).

**"is active"**: old translation sets are retained (for revert / comparison).
`is_active` is a **per-set** flag marking the set currently fed to embedding; all
its parts inherit it. **At most one** active set per document (partial unique
index). **Zero** active is also a valid state: it means the document is currently
not translated — treated the same, for RAG purposes, as a document that hasn't been
translated yet. Embedding and stats treat zero-active as normal, not an error.

**Transactions & idempotence**: each translation set — the set row *and all its
parts* — is committed in **one transaction**, even within a batch run. A set is
therefore all-or-nothing (never a partially-written set), an interrupted batch never
loses completed sets, and a crash mid-batch leaves a consistent DB. Idempotence is
*not* built into the engine; it comes from **how you phrase the selection**. Typical
queries are naturally idempotent, e.g. "translate all untranslated documents",
"…all translated more than 100 days ago", "…all translated with an older prompt
version". Choosing correct, convergent selection filters is the **user's
responsibility**.

**Git gating**: a translation may only be **committed to the DB when the repo is
fully committed** (so `git_rev` is meaningful). Two escape hatches that do *not*
require a clean repo:
- **Inspection mode**: write the translation result to a text file for review.
- **Dry-run mode**: run the translation but do not store it in the DB.

## Translator API

The actual conversion work is done by a **`Translator`** — an abstract class with a
single method:

```python
class Translator(ABC):
    """Converts one document into a list of translations. Must be STATELESS —
    an instance holds no per-document state and can be reused/shared freely."""

    @abstractmethod
    def translate(self, document: "Document") -> "list[Translation]":
        """Translate one document into zero or more Markdown translations."""
```

**`Document`** — everything a translator needs about one input:

```python
@dataclass(frozen=True)
class Document:
    doc_id: str                       # the source document id
    sub_doc_id: list[str]             # path within the document (see below)
    content: bytes                    # raw bytes to translate
    stable_metadata: dict[str, Any]   # from the documents row (or a synthetic doc)
    transient_metadata: dict[str, Any]

    @property
    def mime_type(self) -> "str | None":
        """Convenience accessor — the mime type lives in the metadata, not as a
        separate field. Equivalent to stable_metadata.get("mime-type")."""
        return self.stable_metadata.get("mime-type")
```

**`Translation`** — one produced Markdown part:

```python
@dataclass(frozen=True)
class Translation:
    doc_id: str                       # MUST equal the source document's doc_id
    sub_doc_id: list[str]             # identifies this part within the document
    text: str                         # the Markdown result
    stable_metadata: dict[str, Any]   # per-part metadata
    # NOTE: no transient metadata — translations carry stable metadata only.
```

**`sub_doc_id`** is a list identifying a part *within* a document, so a translator
can recurse (e.g. an email whose attachment is itself an email). For a document
coming straight from the extraction DB it is the **empty list** `[]`. Translators
extend it as they descend (email body → `[..., "body"]`; attachment *n* →
`[..., "attachment:n"]`). It is stored **losslessly** on `translation_parts` as a
JSONB list (`sub_doc_id`), with a generated canonical `part_key` (`json(...)`) for
the per-set uniqueness constraint. A `"/"`-joined form may be used for display only.

## What the pipeline does (not the translator)

The translator is pure conversion; the **translation pipeline** owns all
bookkeeping and DB work:

- Selects source documents (filters), builds a `Document` for each (with
  `sub_doc_id = []`), and calls the chosen translator's `translate`.
- **Enforces `Translation.doc_id == source doc_id`** for every returned
  translation — individual translators are *not* trusted to get this right; a
  mismatch is rejected. (They translate content; identity is the pipeline's.)
- **Enforces unique `sub_doc_id`s within the returned list** (i.e. within the one
  translation set) — a translator that emits two translations with the same
  `sub_doc_id` is rejected before any DB write, not left to fail on the
  `(set_id, part_key)` unique index.
- Writes the returned list as one **translation set** + its **parts** in a single
  transaction, deriving each `part_key` from the translation's `sub_doc_id`, storing
  each `text` as a Markdown blob, and attaching run-level metadata (translator name,
  git rev, timestamp, …) to the set.
- Because the translator is stateless and side-effect-free, dry-run/inspection
  modes just skip the DB write.

## Selecting a translator

The pipeline takes a **`--translator <name>`** argument selecting which translator
subclass to use. Translators live in a subpackage **`translate.translators`**, **one
file per class**; the **file name is the selectable `--translator` value**. Each
file exposes a module-level instance named `translator` (e.g. `translator =
Dispatch()`), which is what the pipeline loads and calls. Default: **`dispatch`**.

## Built-in translators (`translate/translators/`)

- **`dispatch`** (`Dispatch`, default) — routes by `mime_type`:
  - RFC 5322 email (`message/rfc822`) → `Email`;
  - text / HTML / Markdown (`text/plain`, `text/html`, `text/markdown`,
    `application/xhtml+xml`) → `Text`;
  - PDF (`application/pdf`) → `PDF`;
  - Word OOXML (`…wordprocessingml.document`, `…wordprocessingml.template`) → `Docx`;
  - Excel OOXML (`…spreadsheetml.sheet`, `…spreadsheetml.template`) → `Xlsx`;
  - cryptographic signature parts (`application/x-pkcs7-signature`,
    `application/pkcs7-signature`, `application/pgp-signature`) → `Ignore`.
  (Unknown types produce no translation, with a warning — TBD as more types are
  added.)
- **`email`** (`Email`):
  - Splits the message into the **body** (main text, no attachments) and the
    **attachments**.
  - Body → plaintext/Markdown: if already plain text, keep it; if HTML, convert with
    an HTML→Markdown library. Prepend the most important **decoded** header fields
    (e.g. `Subject: Bääh`, not the raw `Subject: =?utf-8?…`). Emits one
    `Translation` with `sub_doc_id = document.sub_doc_id + ["body"]`.
  - For each attachment, builds a **synthetic `Document`** (same `doc_id`,
    `sub_doc_id = document.sub_doc_id + ["attachment:<n>"]`, the attachment's bytes
    + mime-type) and calls **`Dispatch`** to translate it (so attachments of any
    supported type — including nested emails/PDFs — are handled uniformly).
  - Returns the body translation plus all attachment translations.
- **`text`** (`Text`): if the content is plain text, take it; if HTML, convert to
  Markdown; if Markdown, take it. Nothing further. One `Translation`, same
  `sub_doc_id`.
- **`pdf`** (`PDF`): extract the text from the PDF and return it as one
  `Translation`.
- **`docx`** (`Docx`): convert a Word OOXML document (`.docx`/`.dotx`) to Markdown
  via `mammoth` (docx→semantic HTML, mapping headings/lists/tables/bold) plus the
  shared HTML→Markdown helper. One `Translation`.
- **`xlsx`** (`Xlsx`): convert an Excel OOXML workbook (`.xlsx`/`.xltx`) to
  Markdown via `openpyxl` — one `## <sheet>` section per worksheet, each a Markdown
  table of its cached cell values (formulas are not evaluated). One `Translation`.
- **`ignore`** (`Ignore`): always returns no translations. Used by `dispatch` for
  mime types that carry no indexable content (e.g. S/MIME signature parts) so they
  are dropped silently rather than warned about.

## Helpers

- **HTML → Markdown** conversion, backed by an existing library (markdownify) —
  shared by `Email` and `Text`. Lives in `translate/markdown.py`.
- **`make_translation(document, text, ...)`** — convenience constructor that builds
  a `Translation` from a `Document` and the translated text, copying over `doc_id`,
  `sub_doc_id` and the relevant metadata so translators don't repeat that plumbing.
  Lives with the `Translator`/`Document`/`Translation` definitions in
  `translate/base.py`.
