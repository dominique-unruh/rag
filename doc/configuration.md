# Configuration & Secrets

## Config

- A single central YAML file holds everything except secrets (endpoints, model
  names, chunking, storage paths, per-source settings, …).
- Each extractor has its **own subtree** (keyed by its `source_type`, under the
  top-level `extractors:` map); the framework hands a plugin just that subtree, so
  plugins never parse the global file or see unrelated keys.

## Secrets

- Resolved, in order, from the desktop **keyring**, then an optional **file**, then
  — only when running interactively — an input **prompt** whose answer is cached
  back into the keyring.
- No secret ever lives in the YAML or the repo. Plugins use the **same
  secret-resolution mechanism** as the core (passed into `run`), not the config.

## OAuth2 tokens (email `auth: xoauth2`, e.g. Microsoft 365)

Some IMAP servers no longer accept passwords and require OAuth2 bearer tokens via
SASL `XOAUTH2`.

- A one-time interactive **device-code** login seeds a long-lived **refresh
  token**; every run exchanges it for a short-lived access token used for
  `XOAUTH2`.
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
