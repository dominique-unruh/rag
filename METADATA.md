# Metadata keys

Every extracted document (the `documents` table) carries two JSONB metadata maps:

- **`stable_metadata`** — content-defining. A change bumps the document `version`
  and retriggers the pipeline.
- **`transient_metadata`** — volatile state (folder, read flag, …). Written on
  every extraction but **never** bumps the version.

Some keys are **global** — recognised across all extractors and, for the stable
ones listed below, promoted to indexed *generated columns* on `documents`. Other
keys are **extractor-specific**. Each extractor is free to add its own keys; it
should use the global names for the global concepts.

All timestamps use the shared canonical format (`rag.timestamps.format_timestamp`):
**ISO-8601, normalised to UTC, with the `Z` designator**, e.g. `2026-07-15T09:30:00Z`.

---

## Global stable keys

These are recognised everywhere and exposed as indexed generated columns
(`documents.date`, `documents.title`, `documents.mime_type`) via
`stable_metadata ->> '<key>'`.

| Key         | Column      | Format / type            | Meaning                              |
|-------------|-------------|--------------------------|--------------------------------------|
| `date`      | `date`      | ISO-8601 UTC `…Z` string | The document's primary date/time.    |
| `title`     | `title`     | free text (Unicode)      | Short human-readable title.          |
| `mime-type` | `mime_type` | IANA media type string   | Content type of the source blob.     |

Per-extractor notes:

- **`date`** — email: taken from the `Date` header; if that is missing or
  unparseable, approximated from the first parseable `Received` header. Omitted
  entirely if neither yields a date.
- **`title`** — email: the decoded **Subject** (`title` = Subject).
- **`mime-type`** — email: always `message/rfc822`.

There are no global **transient** keys yet (transient keys are currently
extractor-specific).

---

## Email (`source_type = email`)

### `stable_metadata`

| Key           | Global? | Format                                   | Meaning |
|---------------|---------|------------------------------------------|---------|
| `date`        | global  | ISO-8601 UTC `…Z`                        | Message date (Date header, else first Received). |
| `title`       | global  | free text (Unicode, RFC 2047-decoded)    | The decoded **Subject**. |
| `mime-type`   | global  | `message/rfc822`                         | Always the raw RFC 5322 message. |
| `sender`      | email   | `NAME <EMAIL>` (NAME optional)           | Cleaned `From`. |
| `recipients`  | email   | comma-separated `NAME <EMAIL>` list      | Cleaned `To` + `Cc`, merged. |
| `references`  | email   | JSON list of bare message-ids            | The `References` header (thread ancestry). Omitted if absent. |
| `in-reply-to` | email   | JSON list of bare message-ids            | The `In-Reply-To` header. Omitted if absent. |

Message-ids in `references` / `in-reply-to` are stored **without** angle brackets,
matching the `doc_id` form (`<message-id>`), so they reference other messages.

The blob is the full raw email (RFC 5322, headers included).

### `transient_metadata`

| Key       | Format          | Meaning |
|-----------|-----------------|---------|
| `account` | string          | The configured `account-id` the message currently lives in. |
| `folder`  | string          | The IMAP folder (server wire form, e.g. modified UTF-7 like `Entw&APw-rfe`). |
| `read`    | boolean         | Whether the message is currently marked read (`\Seen`). |

Moving a message between accounts/folders or reading it updates the transient
metadata **without** bumping the version.

---

## Synthetic sub-documents (email attachments)

When the email translator descends into attachments it builds *synthetic*
`Document`s (not stored in the `documents` table) whose `stable_metadata` is:

| Key         | Format                | Meaning |
|-------------|-----------------------|---------|
| `mime-type` | IANA media type       | The attachment's content type. |
| `filename`  | free text (`""` if none) | The attachment filename. |

---

## Translation part metadata

Translation parts (`translation_parts.metadata`) do **not** have a fixed schema.
`make_translation` seeds a part's metadata from the source document's stable
metadata, and individual translators may add their own keys. Translations carry
**stable** metadata only (no transient).
