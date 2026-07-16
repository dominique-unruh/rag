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
import uuid
import webbrowser
from html import escape as _h
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from ..blobs import BlobStore
from ..config import Config

# Unique id for this server process. The reloader restarts the process on a
# source change, giving a new id — the main page polls it and reloads when it
# changes, so the browser follows the auto-reload.
_BOOT_ID: str = uuid.uuid4().hex

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

# The single-page app HTML, loaded from a sibling file (kept out of Python so the
# HTML/JS is editable without string-escaping headaches).
_INDEX_HTML: str = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


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


def _source_url(
    source_type: str, doc_id: str, stable: dict[str, Any], transient: dict[str, Any]
) -> str | None:
    """Ask the owning extractor for a source URL for this document, or ``None``.

    Loads the plugin for ``source_type`` and calls its ``source_url``; any
    failure (unknown plugin, plugin error) degrades gracefully to no link.
    """
    # Imported lazily to keep the module import light and avoid cycles.
    from ..extract.discovery import load_plugin

    try:
        return load_plugin(source_type)().source_url(doc_id, stable, transient)
    except Exception:
        # A browser must never break because a plugin's URL logic raised.
        return None


def _kv_table_html(data: dict[str, Any]) -> str:
    """Render a metadata dict as a small HTML key/value table."""
    if not data:
        return "<p><em>(none)</em></p>"
    rows = "".join(
        f"<tr><th>{_h(str(k))}</th><td>{_h(v if isinstance(v, str) else json.dumps(v))}</td></tr>"
        for k, v in data.items()
    )
    return f"<table class='meta'>{rows}</table>"


def _translation_page_html(
    source_type: str,
    doc_id: str,
    translator_name: str,
    source_url: str | None,
    translations: list[Any],
) -> str:
    """Render the translation dry-run page: source link + one foldable section per part."""
    # A link to the original source, if the extractor provides one.
    source_link = (
        f"<p><a href='{_h(source_url)}'>source: {_h(source_url)}</a></p>"
        if source_url
        else "<p class='muted'>(no source link)</p>"
    )
    if not translations:
        sections = "<p class='muted'>(translator produced no translations)</p>"
    else:
        parts: list[str] = []
        for t in translations:
            sub = "/".join(t.sub_doc_id) or "(root)"
            parts.append(
                "<details open><summary>" + _h(sub) + "</summary>"
                + "<h4>sub_doc_id</h4><code>" + _h(json.dumps(t.sub_doc_id)) + "</code>"
                + "<h4>metadata</h4>" + _kv_table_html(t.stable_metadata)
                + "<h4>debug_info</h4><pre>" + _h(t.debug_info or "(none)") + "</pre>"
                + "<h4>content</h4><pre class='content'>" + _h(t.text) + "</pre>"
                + "</details>"
            )
        sections = "\n".join(parts)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>translate: {_h(doc_id)}</title>"
        "<style>"
        "body{font:14px/1.5 system-ui,sans-serif;margin:16px;max-width:1000px}"
        ".muted{color:#888}"
        "table.meta{border-collapse:collapse;margin:4px 0}"
        "table.meta th{text-align:left;padding:2px 10px 2px 0;color:#555;vertical-align:top}"
        "table.meta td{padding:2px 0}"
        "details{border:1px solid #ddd;border-radius:5px;padding:6px 10px;margin:8px 0}"
        "summary{font-weight:600;cursor:pointer}"
        "h4{margin:8px 0 2px;font-size:12px;color:#666;text-transform:uppercase}"
        "pre{background:#f7f7f9;border:1px solid #e2e2e6;padding:8px;overflow:auto;white-space:pre-wrap}"
        "pre.content{max-height:none}"
        "code{background:#f7f7f9;padding:1px 4px}"
        "</style></head><body>"
        f"<h1>Translation dry-run</h1>"
        f"<p><strong>{_h(source_type)}:{_h(doc_id)}</strong> — translator: <code>{_h(translator_name)}</code></p>"
        f"{source_link}{sections}</body></html>"
    )


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

    @app.get("/api/bootid")
    def api_bootid() -> Response:
        """Return this server process's boot id (for main-page auto-reload)."""
        return Response(_BOOT_ID, mimetype="text/plain")

    @app.get("/api/documents")
    def api_documents() -> Response:
        """Return the document list (summary fields) as JSON, newest first."""
        rows = database.query(
            "SELECT source_type, doc_id, version, updated_at, date, mime_type, "
            "json(stable_metadata) AS stable FROM documents "
            "ORDER BY updated_at DESC"
        )
        return jsonify([_display_row(r) for r in rows])

    @app.get("/api/document")
    def api_document() -> Response:
        """Return one document's full metadata (stable + transient) as JSON."""
        row = fetch_doc()
        if row is None:
            return _not_found()
        stable = json.loads(row["stable"])
        transient = json.loads(row["transient"])
        # Translation sets that exist for this document (for the cross-link list).
        tsets = database.query(
            "SELECT id, source_version, translated_at, is_active "
            "FROM translation_sets WHERE source_type = ? AND doc_id = ? "
            "ORDER BY translated_at DESC, id DESC",
            (row["source_type"], row["doc_id"]),
        )
        return jsonify(
            {
                "source_type": row["source_type"],
                "doc_id": row["doc_id"],
                "version": row["version"],
                "blob_hash": row["blob_hash"],
                "updated_at": row["updated_at"],
                "stable": stable,
                "transient": transient,
                # Link to the original source, if the owning extractor provides one.
                "source_url": _source_url(row["source_type"], row["doc_id"], stable, transient),
                "translations": [
                    {
                        "id": t["id"],
                        "source_version": t["source_version"],
                        "translated_at": t["translated_at"],
                        "is_active": bool(t["is_active"]),
                    }
                    for t in tsets
                ],
            }
        )

    @app.get("/api/translations")
    def api_translations() -> Response:
        """Return the translation-set list (summary fields) as JSON, newest first."""
        rows = database.query(
            "SELECT ts.id, ts.source_type, ts.doc_id, ts.source_version, "
            "ts.translated_at, ts.git_rev, ts.is_active, json(ts.metadata) AS meta, "
            "(SELECT COUNT(*) FROM translation_parts tp WHERE tp.set_id = ts.id) AS parts "
            "FROM translation_sets ts ORDER BY ts.translated_at DESC, ts.id DESC"
        )
        return jsonify(
            [
                {
                    "id": r["id"],
                    "source_type": r["source_type"],
                    "doc_id": r["doc_id"],
                    "source_version": r["source_version"],
                    "translated_at": r["translated_at"],
                    "is_active": bool(r["is_active"]),
                    "translator": json.loads(r["meta"]).get("translator", ""),
                    "parts": r["parts"],
                }
                for r in rows
            ]
        )

    @app.get("/api/translation")
    def api_translation() -> Response:
        """Return one translation set: its fields + parts (with bounded content)."""
        set_id = request.args.get("set_id", "")
        sets = database.query(
            "SELECT id, source_type, doc_id, source_version, translated_at, git_rev, "
            "is_active, json(metadata) AS meta FROM translation_sets WHERE id = ?",
            (set_id,),
        )
        if not sets:
            return _not_found("translation set not found")
        srow = sets[0]
        part_rows = database.query(
            "SELECT part_key, json(sub_doc_id) AS sub, json(metadata) AS meta, blob_hash "
            "FROM translation_parts WHERE set_id = ? ORDER BY part_key",
            (set_id,),
        )
        parts = []
        for p in part_rows:
            # Bounded text preview of the translated Markdown blob.
            try:
                data = blobs.get(p["blob_hash"])
                text = data[:_PREVIEW_LIMIT].decode("utf-8", errors="replace")
                if len(data) > _PREVIEW_LIMIT:
                    text += "\n\n[... truncated ...]"
            except FileNotFoundError:
                text = "(blob missing)"
            parts.append(
                {
                    "part_key": p["part_key"],
                    "sub_doc_id": json.loads(p["sub"]),
                    "metadata": json.loads(p["meta"]),
                    "text": text,
                }
            )
        return jsonify(
            {
                "id": srow["id"],
                "source_type": srow["source_type"],
                "doc_id": srow["doc_id"],
                "source_version": srow["source_version"],
                "translated_at": srow["translated_at"],
                "git_rev": srow["git_rev"],
                "is_active": bool(srow["is_active"]),
                "translator": json.loads(srow["meta"]).get("translator", ""),
                "parts": parts,
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

    @app.get("/translate")
    def translate_preview() -> Response:
        """Full HTML page: a translation dry-run of one document (opened in a new tab)."""
        row = fetch_doc()
        if row is None:
            return _not_found()
        stable = json.loads(row["stable"])
        transient = json.loads(row["transient"])
        translator_name = request.args.get("translator", "dispatch")
        # Lazy imports: the translate package is only needed for this endpoint.
        from ..translate.base import Document
        from ..translate.discovery import load_translator

        try:
            content = blobs.get(row["blob_hash"])
            translator = load_translator(translator_name)
            document = Document(row["doc_id"], [], content, stable, transient)
            translations = translator.translate(document)  # dry-run: nothing stored
        except Exception as exc:  # surface any translator/loader error in the page
            body = f"<h1>Translation error</h1><pre>{_h(str(exc))}</pre>"
            return Response(f"<!doctype html><meta charset=utf-8>{body}",
                            mimetype="text/html", status=500)
        source_url = _source_url(row["source_type"], row["doc_id"], stable, transient)
        html = _translation_page_html(
            row["source_type"], row["doc_id"], translator_name, source_url, translations
        )
        return Response(html, mimetype="text/html; charset=utf-8")

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
    # by WERKZEUG_RUN_MAIN, re-spawned on every source change).
    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if is_reloader_child:
        # The child (re)binds the port on start and each reload.
        print(f"db-browser serving at {url} (Ctrl-C to stop, auto-reloads on source changes)")
    elif open_browser:
        # Open the browser ONCE, from the supervisor — it is not re-run on reload,
        # so reloads do not spawn new windows. Delay lets the child bind the port.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        # Werkzeug's file-watching auto-reload is always enabled.
        app.run(host=host, port=port, threaded=True, use_reloader=True)
    finally:
        database.close()

