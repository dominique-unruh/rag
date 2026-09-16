# Configuration & Secrets

## Config

- A single central YAML file holds everything except secrets (endpoints, model
  names, chunking, storage paths, per-source settings, …).
- Each extractor has its **own subtree** (keyed by its `source_type`, under the
  top-level `extractors:` map); the framework hands a plugin just that subtree, so
  plugins never parse the global file or see unrelated keys.

### `extractors.documents` (filesystem)

A `folders` list plus two optional plugin-wide keys (see
[extraction.md](extraction.md) for behaviour):

- `folders` — one entry per on-disk folder, each `{folder-id, path, description}`
  plus an optional `sourcelink`. `folder-id` is short, stable and unique (it
  prefixes doc-ids); `path` is the directory to walk (`~` expanded). `sourcelink`
  (`metadata` (default) / `local` / `obsidian`) selects how the folder's documents
  produce their `source_url` link — see [extraction.md](extraction.md).
- `max_file_size` — optional per-file byte cap; files above it are skipped with an
  error. Default 100 MiB.
- `exclude` — optional list of extra basename globs, **added to** the built-in
  defaults (`.*`, `*~`, `*.swp`, `*.tmp`, `*.part`, `#*#`).

Mime detection needs **libmagic** at runtime (the `python-magic` binding is a
declared dependency; also install the system library, e.g. `apt install
libmagic1`). This plugin uses no secrets.

## Secrets

- Resolved, in order, from the desktop **keyring**, then an optional **file**, then
  — only when running interactively — an input **prompt** whose answer is cached
  back into the keyring.
- No secret ever lives in the YAML or the repo. Plugins use the **same
  secret-resolution mechanism** as the core (passed into `run`), not the config.

## OAuth2 tokens (email `auth: xoauth2`, e.g. Microsoft 365)

Some IMAP servers no longer accept passwords and require OAuth2 bearer tokens via
SASL `XOAUTH2`.

- A one-time interactive login seeds a long-lived **refresh token**; every run
  exchanges it for a short-lived access token used for `XOAUTH2`. The login uses
  the **authorization-code + PKCE loopback** flow (opens the system browser,
  captures the redirect on a temporary `http://localhost:<port>` listener) — this
  works even on tenants whose Conditional Access **blocks the device-code flow**
  ("authentication flow restricted by your admin"). The **device-code** flow is
  kept as a fallback for headless hosts with no launchable browser.
- The access token is ephemeral (in-memory only); the refresh token is persisted —
  and rewritten whenever the server rotates it — as one file per account under the
  email plugin's `secrets/` state subdirectory (not the keyring, since it is
  mutable rotating state, and not the YAML).
- Defaults use Thunderbird's public client id / `common` tenant / the
  `outlook.office.com` IMAP scope, overridable per account via an `oauth:` subtree
  (`tenant`, `client_id`, `scopes`, `client_secret`).
- Headless runs cannot do the initial login and fail with an actionable message
  telling the user to run it once in a terminal.

## Timestamp parsing utility

A general timestamp-parsing utility lives in the **core codebase** (not in any
plugin), so every module and plugin parses times the same way. It accepts:
- absolute forms (ISO 8601 dates / datetimes), and
- convenience relative forms, e.g. `1d ago`, `2h ago`, `30m ago`, `1w ago`, and
  keywords like `now` / `today` / `yesterday`.

It is used wherever the CLI or a plugin takes a time (e.g. the email plugin's
`since` / `until`, `translate`'s "older than" filters), giving one consistent
syntax across the whole tool.
