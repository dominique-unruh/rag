"""db-browser: a local web UI for browsing the extraction database (PLAN §8).

Built on Flask so its development server provides auto-reload ("watch") for free.
The left navbar lists tools; currently one, "Extraction", which shows a split
view: the top half lists ``documents`` rows, the bottom half shows the selected
document's full metadata plus a content preview and a download button (using the
document's mime-type when known).

The database connection is opened lazily and **auto-released after 5 seconds of
inactivity** (matching the project's idle-release policy, PLAN §9), so a running
browser does not hold the DB open between clicks.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from ..blobs import BlobStore
from ..config import Config

# Seconds of inactivity after which the shared DB connection is closed.
_IDLE_SECONDS: float = 5.0

# Extra mime -> extension mappings for download filenames where the stdlib
# ``mimetypes`` table is incomplete.
_EXTRA_EXTENSIONS: dict[str, str] = {
    "message/rfc822": ".eml",
    "text/markdown": ".md",
}

# Max bytes of a document rendered inline as a text preview (avoids shipping huge
# blobs into the browser just to look at them).
_PREVIEW_LIMIT: int = 200_000


class IdleDatabase:
    """A read-only SQLite connection that closes itself after inactivity.

    The connection is (re)opened on demand and closed by a background reaper once
    it has been idle for ``idle_seconds``. All access is serialised through a lock
    so the reaper and request threads never touch the connection concurrently.
    """

    def __init__(self, db_path: Path, idle_seconds: float = _IDLE_SECONDS) -> None:
        """Start managing the database at ``db_path`` with the given idle timeout."""
        # Path to the metadata database (opened read-only).
        self._path: Path = db_path
        # How long the connection may sit unused before being closed.
        self._idle: float = idle_seconds
        # How often the reaper checks for idleness (a fraction of the window, but
        # at most once a second and never absurdly fast).
        self._poll: float = max(0.05, min(1.0, idle_seconds / 2))
        # Guards ``_conn`` and ``_last_used`` across request/reaper threads.
        self._lock: threading.Lock = threading.Lock()
        # The live connection, or None when currently closed.
        self._conn: sqlite3.Connection | None = None
        # Monotonic timestamp of the last query (for idle detection).
        self._last_used: float = 0.0
        # Signals the reaper thread to stop on shutdown.
        self._stop: threading.Event = threading.Event()
        # Background thread that closes the connection when idle.
        self._reaper: threading.Thread = threading.Thread(
            target=self._reap_loop, name="dbbrowser-idle", daemon=True
        )
        self._reaper.start()

    def _open(self) -> sqlite3.Connection:
        """Open a fresh read-only connection to the database."""
        # ``mode=ro`` opens read-only; check_same_thread=False because the reaper
        # thread may close a connection opened by a request thread (always under
        # the lock, so this is safe).
        conn = sqlite3.connect(
            f"file:{self._path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return conn

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Run ``sql`` and return all rows, (re)opening the connection as needed."""
        with self._lock:
            # Reopen if the reaper closed it since the last request.
            if self._conn is None:
                self._conn = self._open()
            # Mark activity so the reaper's idle clock restarts.
            self._last_used = time.monotonic()
            return self._conn.execute(sql, params).fetchall()

    def _reap_loop(self) -> None:
        """Background loop: close the connection once it has been idle too long."""
        # Poll a little faster than the idle window so closing is timely.
        while not self._stop.wait(self._poll):
            with self._lock:
                idle_for = time.monotonic() - self._last_used
                if self._conn is not None and idle_for >= self._idle:
                    # No activity within the window — release the connection.
                    self._conn.close()
                    self._conn = None

    def close(self) -> None:
        """Stop the reaper and close any open connection (called on shutdown)."""
        self._stop.set()
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _display_row(row: sqlite3.Row) -> dict[str, Any]:
    """Project a ``documents`` row into the summary fields shown in the top list."""
    stable = json.loads(row["stable"])
    # A single "title" column that works for notes (title) and email (subject).
    title = stable.get("title") or stable.get("subject") or ""
    return {
        "source_type": row["source_type"],
        "doc_id": row["doc_id"],
        "version": row["version"],
        "date": row["date"] or "",
        "mime_type": row["mime_type"] or "",
        "updated_at": row["updated_at"],
        "title": title,
        "sender": stable.get("sender", ""),
    }


def _download_name(doc_id: str, mime: str | None) -> str:
    """Return a safe download filename for ``doc_id`` with a mime-derived extension."""
    # Keep only filename-safe characters from the (possibly URL/@-laden) doc_id.
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in doc_id) or "document"
    # Choose an extension from our overrides first, then the stdlib table.
    ext = ""
    if mime:
        ext = _EXTRA_EXTENSIONS.get(mime) or (mimetypes.guess_extension(mime) or "")
    # Avoid doubling an extension the id already carries.
    return safe if ext and safe.endswith(ext) else safe + ext


def _create_app(database: IdleDatabase, blobs: BlobStore) -> Flask:
    """Build the Flask app with the extraction-browser routes.

    ``database``/``blobs`` are captured by the route closures; keeping them out of
    module globals lets the app be created fresh (e.g. by the reloader child).
    """
    app = Flask(__name__)

    def fetch_doc() -> sqlite3.Row | None:
        """Fetch the document identified by the request's source_type/doc_id args."""
        source_type = request.args.get("source_type", "")
        doc_id = request.args.get("doc_id", "")
        rows = database.query(
            "SELECT source_type, doc_id, version, blob_hash, updated_at, "
            "json(stable_metadata) AS stable, json(transient_metadata) AS transient "
            "FROM documents WHERE source_type = ? AND doc_id = ?",
            (source_type, doc_id),
        )
        return rows[0] if rows else None

    @app.get("/")
    def index() -> Response:
        """Serve the single-page app."""
        return Response(_INDEX_HTML, mimetype="text/html")

    @app.get("/api/documents")
    def api_documents() -> Response:
        """Return the document list (summary fields) as JSON, newest first."""
        rows = database.query(
            "SELECT source_type, doc_id, version, updated_at, date, mime_type, "
            "json(stable_metadata) AS stable FROM documents "
            "ORDER BY date DESC, updated_at DESC"
        )
        return jsonify([_display_row(r) for r in rows])

    @app.get("/api/document")
    def api_document() -> Response:
        """Return one document's full metadata (stable + transient) as JSON."""
        row = fetch_doc()
        if row is None:
            return _not_found()
        return jsonify(
            {
                "source_type": row["source_type"],
                "doc_id": row["doc_id"],
                "version": row["version"],
                "blob_hash": row["blob_hash"],
                "updated_at": row["updated_at"],
                "stable": json.loads(row["stable"]),
                "transient": json.loads(row["transient"]),
            }
        )

    @app.get("/api/content")
    def api_content() -> Response:
        """Serve a document's blob: inline, as a bounded text preview, or download."""
        row = fetch_doc()
        if row is None:
            return _not_found()
        try:
            data = blobs.get(row["blob_hash"])
        except FileNotFoundError:
            return _not_found("blob not found")
        mime = json.loads(row["stable"]).get("mime-type") or "application/octet-stream"

        if request.args.get("preview") == "1":
            # Bounded, decoded text preview for the UI.
            text = data[:_PREVIEW_LIMIT].decode("utf-8", errors="replace")
            if len(data) > _PREVIEW_LIMIT:
                text += "\n\n[... truncated ...]"
            return Response(text, mimetype="text/plain; charset=utf-8")

        headers: dict[str, str] = {}
        if request.args.get("download") == "1":
            # Force a download with a sensible, mime-derived filename.
            filename = _download_name(row["doc_id"], mime)
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        # Otherwise serve inline with the document's own mime-type.
        return Response(data, mimetype=mime, headers=headers)

    def _not_found(message: str = "document not found") -> Response:
        """Return a 404 JSON error."""
        response = jsonify({"error": message})
        response.status_code = 404
        return response

    return app


def serve(
    config: Config, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> None:
    """Run the db-browser web server until interrupted (Ctrl-C).

    Binds to ``host``/``port`` (localhost by default), optionally opens a browser,
    and serves the extraction database found under the configured storage dir.
    Flask's reloader is always on: the server restarts on source changes.
    """
    storage = config.storage_dir
    database = IdleDatabase(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    app = _create_app(database, blobs)
    url = f"http://{host}:{port}/"
    # The reloader runs a parent supervisor plus the actual serving child (marked
    # by WERKZEUG_RUN_MAIN); print/open the browser only in the serving child.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print(f"db-browser serving at {url} (Ctrl-C to stop, auto-reloads on source changes)")
        if open_browser:
            # Open shortly after the server starts accepting connections.
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        # Werkzeug's file-watching auto-reload is always enabled.
        app.run(host=host, port=port, threaded=True, use_reloader=True)
    finally:
        database.close()


# The single-page app: left navbar + split (list over detail) Extraction view.
_INDEX_HTML: str = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>rag db-browser</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.4 system-ui, sans-serif; height: 100vh; display: flex; }
  nav { width: 160px; background: #1f2430; color: #cdd3df; flex: 0 0 160px; }
  nav h1 { font-size: 14px; padding: 12px; margin: 0; color: #fff; border-bottom: 1px solid #333b4d; }
  nav a { display: block; padding: 10px 12px; color: #cdd3df; text-decoration: none; cursor: pointer; }
  nav a.active, nav a:hover { background: #2b3242; color: #fff; }
  main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #list { flex: 1 1 45%; overflow: auto; border-bottom: 2px solid #ccc; }
  #detail { flex: 1 1 55%; overflow: auto; padding: 10px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; max-width: 340px; }
  th { position: sticky; top: 0; background: #f5f6f8; }
  #list tr { cursor: pointer; }
  #list tr.sel { background: #dce7ff; }
  #detail h2 { font-size: 14px; margin: 6px 0; }
  .meta td:first-child { color: #555; width: 160px; font-weight: 600; }
  pre { background: #f7f7f9; border: 1px solid #e2e2e6; padding: 8px; overflow: auto; max-height: 40vh; }
  .bar { margin: 8px 0; }
  a.btn { background: #2b6cff; color: #fff; padding: 5px 10px; border-radius: 4px; text-decoration: none; }
  .muted { color: #888; }
  #ctxmenu { position: fixed; z-index: 10; background: #fff; border: 1px solid #bbb;
             box-shadow: 0 2px 8px rgba(0,0,0,.2); display: none; }
  #ctxmenu div { padding: 6px 14px; cursor: pointer; white-space: nowrap; }
  #ctxmenu div:hover { background: #2b6cff; color: #fff; }
</style>
</head>
<body>
<nav>
  <h1>db-browser</h1>
  <a class="active" data-tool="extraction">Extraction</a>
</nav>
<main>
  <div id="list"><p class="muted" style="padding:10px">Loading…</p></div>
  <div id="detail"><p class="muted">Select a row above.</p></div>
</main>
<div id="ctxmenu"><div data-act="copysql">Copy SQL</div></div>
<script>
const listEl = document.getElementById('list');
const detailEl = document.getElementById('detail');
const menuEl = document.getElementById('ctxmenu');

// Current selectable rows and the index of the selected one (-1 = none).
let rowEls = [];
let selIndex = -1;
// The row the context menu currently targets ({st, id}) or null.
let menuTarget = null;

// Quote a value as a SQLite string literal (single quotes doubled).
function sqlLit(v) { return "'" + String(v).replace(/'/g, "''") + "'"; }

// Build a `FROM ... WHERE ...` clause selecting exactly one document row, so
// `SELECT * <clause>` returns that row.
function rowSql(st, id) {
  return `FROM documents WHERE source_type = ${sqlLit(st)} AND doc_id = ${sqlLit(id)}`;
}

// Hide the context menu.
function hideMenu() { menuEl.style.display = 'none'; menuTarget = null; }

// Copy text to the clipboard, with a fallback for non-secure contexts.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); ta.remove();
  }
}

// Dismiss the menu on any outside interaction.
document.addEventListener('click', hideMenu);
document.addEventListener('scroll', hideMenu, true);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideMenu(); });

// "Copy SQL" action.
menuEl.querySelector('[data-act="copysql"]').onclick = (e) => {
  e.stopPropagation();
  if (menuTarget) copyText(rowSql(menuTarget.st, menuTarget.id));
  hideMenu();
};

// Escape text for safe insertion into HTML.
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Reflect the current selection in the URL hash so the page is deep-linkable.
function updateHash(st, id) {
  const h = '#' + new URLSearchParams({ st, id }).toString();
  // replaceState avoids piling up history entries as the selection moves.
  if (location.hash !== h) history.replaceState(null, '', h);
}

// Select the row at index i (clamped): highlight it, scroll it into view, load
// its detail and update the URL. Used by clicks, arrow keys and deep links.
function selectRow(i) {
  if (!rowEls.length) return;
  i = Math.max(0, Math.min(rowEls.length - 1, i));
  if (selIndex >= 0 && rowEls[selIndex]) rowEls[selIndex].classList.remove('sel');
  selIndex = i;
  const tr = rowEls[i];
  tr.classList.add('sel');
  tr.scrollIntoView({ block: 'nearest' });
  showDetail(tr.dataset.st, tr.dataset.id);
  updateHash(tr.dataset.st, tr.dataset.id);
}

// Select the row named by the URL hash (#st=..&id=..), if present and found.
function applyHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const st = p.get('st'), id = p.get('id');
  if (!st || !id) return false;
  const idx = rowEls.findIndex(tr => tr.dataset.st === st && tr.dataset.id === id);
  if (idx >= 0) { selectRow(idx); return true; }
  return false;
}

// Follow the hash when the user pastes/edits a deep link.
window.addEventListener('hashchange', applyHash);

// Arrow up/down move the selection (when the list has rows).
document.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  if (!rowEls.length) return;
  e.preventDefault();
  // Start from the top on first keypress if nothing is selected yet.
  const base = selIndex < 0 ? (e.key === 'ArrowDown' ? -1 : 0) : selIndex;
  selectRow(base + (e.key === 'ArrowDown' ? 1 : -1));
});

// Load the document list into the top half.
async function loadList() {
  const rows = await (await fetch('/api/documents')).json();
  if (!rows.length) { listEl.innerHTML = '<p class="muted" style="padding:10px">No documents.</p>'; return; }
  let html = '<table><thead><tr><th>date</th><th>source</th><th>title</th>'
           + '<th>sender</th><th>mime</th><th>v</th></tr></thead><tbody>';
  for (const r of rows) {
    html += `<tr data-st="${esc(r.source_type)}" data-id="${esc(r.doc_id)}">`
          + `<td>${esc(r.date)}</td><td>${esc(r.source_type)}</td>`
          + `<td>${esc(r.title)}</td><td>${esc(r.sender)}</td>`
          + `<td>${esc(r.mime_type)}</td><td>${esc(r.version)}</td></tr>`;
  }
  html += '</tbody></table>';
  listEl.innerHTML = html;
  // Cache the rows and wire clicks through the shared selection logic.
  rowEls = Array.from(listEl.querySelectorAll('tr[data-id]'));
  selIndex = -1;
  rowEls.forEach((tr, i) => {
    tr.onclick = () => selectRow(i);
    // (context menu wiring below)
    // Right-click: select the row and open the context menu at the cursor.
    tr.oncontextmenu = (e) => {
      e.preventDefault();
      selectRow(i);
      menuTarget = { st: tr.dataset.st, id: tr.dataset.id };
      menuEl.style.left = e.clientX + 'px';
      menuEl.style.top = e.clientY + 'px';
      menuEl.style.display = 'block';
    };
  });
  // Restore any selection encoded in the URL now that rows exist.
  applyHash();
}

// Render a key/value object as a metadata table.
function metaTable(obj) {
  const keys = Object.keys(obj);
  if (!keys.length) return '<p class="muted">(none)</p>';
  return '<table class="meta">' + keys.map(k =>
    `<tr><td>${esc(k)}</td><td>${esc(typeof obj[k] === 'object' ? JSON.stringify(obj[k]) : obj[k])}</td></tr>`
  ).join('') + '</table>';
}

// Load and show the detail (metadata + preview + download) for one document.
async function showDetail(st, id) {
  const q = `source_type=${encodeURIComponent(st)}&doc_id=${encodeURIComponent(id)}`;
  const doc = await (await fetch('/api/document?' + q)).json();
  const core = { source_type: doc.source_type, doc_id: doc.doc_id,
                 version: doc.version, updated_at: doc.updated_at, blob_hash: doc.blob_hash };
  detailEl.innerHTML =
      '<h2>Document</h2>' + metaTable(core)
    + '<h2>Stable metadata</h2>' + metaTable(doc.stable)
    + '<h2>Transient metadata</h2>' + metaTable(doc.transient)
    + '<h2>Content</h2>'
    + `<div class="bar"><a class="btn" href="/api/content?${q}&download=1">Download</a></div>`
    + '<pre id="preview" class="muted">Loading preview…</pre>';
  const txt = await (await fetch('/api/content?' + q + '&preview=1')).text();
  const pre = document.getElementById('preview');
  pre.classList.remove('muted');
  pre.textContent = txt;
}

loadList();
</script>
</body>
</html>
"""
