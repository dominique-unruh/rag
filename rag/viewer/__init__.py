"""Interactive store viewer — a small Flask app, one item at a time.

Not a static dump: every page loads fresh from the graph store, so it reflects
the current DB. You browse item-by-item — an object (entity), a relation, a
document, or a chunk — with all properties in a table and every reference
rendered as a clickable link that navigates to the next item. A browser-local
history panel records where you've been and lets you jump back.

    python -m rag.viewer [--port 5000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote, urlencode

import yaml
from flask import Flask, request
from werkzeug.exceptions import HTTPException

from .. import cli, config, runtime, stores

app = Flask(__name__)
_CFG: config.Config | None = None
_GRAPH = None        # KuzuPropertyGraphStore (opened once in main)
_GRAPH_DB = None     # underlying kuzu.Database (closed at shutdown)
_DOCS = None         # DocumentStore (canonical documents)


@app.errorhandler(Exception)
def _on_error(exc):
    """Render unexpected errors as a page instead of a bare 500 (local tool)."""
    if isinstance(exc, HTTPException):
        return exc  # keep 404 etc. as-is
    body = (
        "<div class=card><h2 class=title>Error</h2>"
        f"<pre>{html.escape(str(exc))}</pre></div>"
    )
    return _page("error", body), 500


# --- data access ----------------------------------------------------------
# The viewer reads the PropertyGraphStore *interface* (get / get_triplets), so it
# works over Kùzu and is easy to fake in tests. Documents are read from the
# Document Store (canonical), not the graph.

class _GraphView:
    """A per-request snapshot: nodes dict + relations list, from the store API."""

    def __init__(self, store):
        self.nodes = {n.id: n for n in store.get()}
        # Kùzu's Relation.source_id/target_id are unreliable (both come back as the
        # target); take the true endpoints from the triplet's own nodes.
        relations = []
        for src, rel, tgt in store.get_triplets():
            rel.source_id = src.id
            rel.target_id = tgt.id
            relations.append(rel)
        self.relations = relations


import threading
import time

IDLE_RELEASE_SECONDS = 5.0
_lock = threading.RLock()
_last_access = 0.0


def _ensure_stores() -> None:
    """Open the graph + document stores lazily, in the serving process.

    Under `--reload` the reloader parent must NOT open Kùzu (single-writer lock),
    only the worker that handles requests — so we defer opening until first use.
    Every touch bumps `_last_access` so the idle watcher can release the DB locks
    (freeing them for the pipelines) after inactivity, then reopen on demand.
    """
    global _GRAPH, _GRAPH_DB, _DOCS, _last_access
    with _lock:
        _last_access = time.monotonic()
        if _GRAPH is None:
            _GRAPH, _GRAPH_DB = stores.open_graph_store(_CFG)
        if _DOCS is None:
            # check_same_thread=False: the idle watcher (a separate thread) closes it.
            _DOCS = stores.open_document_store(_CFG, check_same_thread=False)


def _release_if_idle() -> None:
    global _GRAPH, _GRAPH_DB, _DOCS
    with _lock:
        if _GRAPH is None and _DOCS is None:
            return
        if time.monotonic() - _last_access < IDLE_RELEASE_SECONDS:
            return
        if _GRAPH_DB is not None:
            try:
                stores.close_graph_db(_GRAPH_DB)
            except Exception:
                pass
        if _DOCS is not None:
            try:
                _DOCS.close()
            except Exception:
                pass
        _GRAPH = _GRAPH_DB = _DOCS = None
        print("(released DB connections after idle)", file=sys.stderr)


def _start_idle_watcher() -> None:
    def loop():
        while True:
            time.sleep(1.0)
            try:
                _release_if_idle()
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True, name="rag-viewer-idle").start()


def _graph() -> _GraphView:
    _ensure_stores()
    return _GraphView(_GRAPH)


def _nodes(graph):
    return graph.nodes  # dict[id -> LabelledNode]


def _relations(graph):
    return graph.relations


def _doc_ids() -> set[str]:
    _ensure_stores()
    return set(_DOCS.want_hashes()) if _DOCS is not None else set()


def _kind_of(graph, item_id: str) -> str:
    # Documents live in the Document Store (canonical); check there first.
    if item_id in _doc_ids():
        return "document"
    node = _nodes(graph).get(item_id)
    if node is not None:
        return "chunk" if getattr(node, "label", "") == "text_chunk" else "entity"
    return "entity"  # unknown reference; try as entity


def _is_entity(graph, node_id: str, node) -> bool:
    return getattr(node, "label", "") != "text_chunk" and node_id not in _doc_ids()


def _display(graph, item_id: str) -> str:
    if item_id in _doc_ids():
        return item_id
    node = _nodes(graph).get(item_id)
    if node is None:
        return item_id
    if getattr(node, "label", "") == "text_chunk":
        text = (getattr(node, "text", "") or "").strip().replace("\n", " ")
        return f"chunk: {text[:40]}…" if text else f"chunk {item_id[:8]}"
    return getattr(node, "name", None) or item_id


# --- URL + HTML helpers ---------------------------------------------------

def _url(kind: str, item_id: str) -> str:
    return "/item?" + urlencode({"kind": kind, "id": item_id})


def _rel_id(r) -> str:
    """Opaque relation id. Relations have no natural id (only the source/label/
    target triple is unique), so pack the triple into one url-safe token."""
    raw = json.dumps([r.source_id, r.label, r.target_id])
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _rel_from_id(graph, rid: str):
    try:
        source, label, target = json.loads(base64.urlsafe_b64decode(rid))
    except Exception:
        return None
    for r in _relations(graph):
        if r.source_id == source and r.label == label and r.target_id == target:
            return r
    return None


def _rel_url(r) -> str:
    return _url("relation", _rel_id(r))


def _link(graph, item_id: str, text: str | None = None) -> str:
    kind = _kind_of(graph, item_id)
    label = html.escape(text or _display(graph, item_id))
    return f'<a href="{html.escape(_url(kind, item_id))}">{label}</a>'


def _linkify_value(graph, value) -> str:
    """Render a property value; if it references a node/document, link it."""
    s = str(value)
    if s in _nodes(graph) or s in _doc_ids():
        return _link(graph, s)
    return html.escape(s)


def _render_md(text: str) -> str:
    """Render chunk text as Markdown (python-markdown). Falls back to plain."""
    try:
        import markdown
    except Exception:
        return f"<pre>{html.escape(text)}</pre>"
    return markdown.markdown(text or "", extensions=["fenced_code", "tables", "sane_lists"])


_DESCRIPTIONS: dict | None = None


def _descriptions() -> dict:
    """Property descriptions: project ./properties.yaml overrides the bundled one."""
    global _DESCRIPTIONS
    if _DESCRIPTIONS is None:
        _DESCRIPTIONS = {}
        for path in (Path("properties.yaml"), Path(__file__).parent.parent / "properties.yaml"):
            if path.exists():
                try:
                    _DESCRIPTIONS = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    _DESCRIPTIONS = {}
                break
    return _DESCRIPTIONS


def _prop_desc(kind: str, key: str) -> str | None:
    d = _descriptions()
    section = d.get(kind, {}) or {}
    common = d.get("common", {}) or {}
    return section.get(key) or common.get(key)


def _try_json(val):
    """Parse a string that holds a JSON object/array; else None (no error)."""
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _json_tree(value, top: bool = True) -> str:
    """Foldable JSON -> HTML using native <details> (interactive, no JS lib)."""
    if isinstance(value, dict):
        if not value:
            return "<span class=jpunct>{}</span>"
        rows = "".join(
            f'<div class=jrow><span class=jkey>{html.escape(str(k))}</span>: '
            f"{_json_tree(v, False)}</div>"
            for k, v in value.items()
        )
        summary = f"{{…}} {len(value)} keys"
        return (f'<details class=jt{" open" if top else ""}><summary>{summary}</summary>'
                f"<div class=jbody>{rows}</div></details>")
    if isinstance(value, list):
        if not value:
            return "<span class=jpunct>[]</span>"
        rows = "".join(f"<div class=jrow>{_json_tree(v, False)}</div>" for v in value)
        summary = f"[…] {len(value)} items"
        return (f'<details class=jt{" open" if top else ""}><summary>{summary}</summary>'
                f"<div class=jbody>{rows}</div></details>")
    if isinstance(value, bool):
        return f"<span class=jbool>{str(value).lower()}</span>"
    if value is None:
        return "<span class=jnull>null</span>"
    if isinstance(value, str):
        return f'<span class=jstr>"{html.escape(value)}"</span>'
    return f"<span class=jnum>{html.escape(str(value))}</span>"


def _copy_btn(value: str) -> str:
    """Unobtrusive copy-to-clipboard button (self-contained; see _COPY_JS)."""
    return (
        '<button class=copy type=button title="copy value" '
        f'data-copy="{html.escape(str(value), quote=True)}">⧉</button>'
    )


# --- path translation -----------------------------------------------------
# Browsers block file:// links from an http page, so a raw path is not clickable.
# Instead, translate known paths to an app URL scheme. Each translator takes
# (cfg, path) and returns a URL or None. First match wins; unknown paths stay
# plain text. Add more translators (other vaults, editors, web mounts) here.

OBSIDIAN_VAULT = "notes"


def _obsidian_translator(cfg, path: str) -> str | None:
    """A markdown file inside the notes vault -> obsidian://open URI.

    Only files under the configured notes directory belong to the vault; any
    other /home/... path is not an Obsidian file and stays unlinked.

    /…/notes/meetings/2026-05-18 TdI Planning.md
      -> obsidian://open?vault=notes&file=meetings%2F2026-05-18%20TdI%20Planning
    """
    if cfg is None or not getattr(cfg.notes, "enabled", False):
        return None
    root = cfg.notes.directory.rstrip("/")
    if not root or not path.startswith(root + "/") or not path.endswith(".md"):
        return None
    rel = path[len(root) + 1:-len(".md")]  # relative to vault root, extension stripped
    return f"obsidian://open?vault={OBSIDIAN_VAULT}&file={quote(rel, safe='')}"


def _messageid_translator(cfg, path: str) -> str | None:
    """An email Message-ID (email doc id) -> mid: URI (RFC 2392).

    Message-IDs are not filesystem paths (no leading /, contain '@'). The id is
    used raw, without URL-encoding.

    CAOCut...+Hqk...@mail.gmail.com -> mid:CAOCut...+Hqk...@mail.gmail.com
    """
    if path.startswith("/") or "@" not in path:
        return None
    return "mid:" + path  # raw Message-ID, not URL-encoded


_PATH_TRANSLATORS = [_obsidian_translator, _messageid_translator]


def _translate_path(path: str) -> str | None:
    for translator in _PATH_TRANSLATORS:
        url = translator(_CFG, path)
        if url:
            return url
    return None


def _path_link(path: str) -> str:
    """Clickable app-scheme link if a translator recognizes the path; else plain."""
    url = _translate_path(path)
    if url:
        return f'<a href="{html.escape(url, quote=True)}">{html.escape(path)}</a>'
    return html.escape(path)  # unrecognized -> not linked


def _prop_th(kind: str, key: str) -> str:
    desc = _prop_desc(kind, key)
    if not desc:
        return f"<th>{html.escape(str(key))}</th>"
    return (f'<th title="{html.escape(str(desc), quote=True)}">{html.escape(str(key))}'
            " <span class=pinfo>ⓘ</span></th>")


def _prop_table(graph, props: dict, linkify: bool = True, kind: str = "") -> str:
    if not props:
        return "<p class=empty>(no properties)</p>"
    rows = []
    for key, val in props.items():
        parsed = val if isinstance(val, (dict, list)) else _try_json(val)
        if isinstance(parsed, (dict, list)):
            raw = val if isinstance(val, str) else json.dumps(val, indent=2, default=str)
            shown = _json_tree(parsed)  # foldable JSON viewer
        elif key == "file_path":
            raw = str(val)
            shown = _path_link(raw)  # translate to an app scheme (e.g. obsidian://)
        elif linkify:
            raw = str(val)
            shown = _linkify_value(graph, val)
        else:
            raw = str(val)
            shown = html.escape(raw)
        rows.append(
            f"<tr>{_prop_th(kind, key)}"
            f"<td>{shown}{_copy_btn(raw)}</td></tr>"
        )
    return f"<table class=props>{''.join(rows)}</table>"


def _relation_row(graph, r, node_id: str) -> str:
    """One relation as: source -[label]-> target, with the relation itself linked."""
    direction = "out" if r.source_id == node_id else "in"
    rel = f'<a href="{html.escape(_rel_url(r))}">{html.escape(r.label)}</a>'
    src = _link(graph, r.source_id)
    tgt = _link(graph, r.target_id)
    return (
        f"<tr><td class=dir>{direction}</td><td>{src}</td>"
        f"<td class=rel>-[{rel}]-&gt;</td><td>{tgt}</td></tr>"
    )


# --- pages ----------------------------------------------------------------

_STYLE = """
body{font:14px/1.55 system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
a{color:#2657c7;text-decoration:none}a:hover{text-decoration:underline}
header{background:#1a1a2e;color:#fff;padding:12px 20px;display:flex;gap:16px;align-items:center}
header h1{margin:0;font-size:16px}header a{color:#9db8ff}
form.search{margin-left:auto;display:flex;gap:8px;align-items:center}
form.search input[name=q]{width:280px;padding:6px 10px;border-radius:6px;border:1px solid #3a3a55;
  background:#2a2a42;color:#fff;font-size:13px}
form.search .sem{color:#aab;font-size:12px;display:flex;gap:4px;align-items:center;cursor:help}
form.search .info{opacity:.7}
.hit{display:flex;gap:10px;align-items:baseline;padding:6px 0;border-bottom:1px solid #f0f0f2}
.hit .score{font-family:ui-monospace,monospace;font-size:11px;color:#999;min-width:34px}
.hit .badge{font-size:10px;text-transform:uppercase;background:#eef1fb;color:#2657c7;
  padding:1px 6px;border-radius:8px;min-width:64px;text-align:center}
.hit .snip{color:#555;font-size:12px}
.layout{display:flex;gap:0;align-items:flex-start}
main{flex:1;padding:20px;min-width:0}
aside{width:230px;background:#fff;border-left:1px solid #e6e6ea;min-height:90vh;padding:12px}
aside h2{font-size:12px;text-transform:uppercase;color:#888;margin:4px 0 8px}
aside ol{list-style:none;margin:0;padding:0;font-size:13px}
aside li{padding:3px 0;border-bottom:1px solid #f0f0f2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:16px 20px;margin-bottom:16px}
.kind{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  background:#eef1fb;color:#2657c7;padding:2px 8px;border-radius:10px}
.kind.entity{background:#eef1fb;color:#2657c7}
.kind.chunk{background:#e7f6ee;color:#1f9d55}
.kind.relation{background:#fbeee1;color:#c2681f}
.kind.document{background:#f2eafb;color:#8a4fbf}
.hb{display:inline-block;min-width:16px;height:16px;line-height:16px;text-align:center;
  border-radius:4px;font-size:10px;font-weight:700;margin-right:6px;color:#fff;padding:0 3px}
.hb.entity{background:#2657c7}.hb.chunk{background:#1f9d55}.hb.relation{background:#c2681f}
.hb.document{background:#8a4fbf}.hb.search{background:#888}.hb.home{background:#aaa}
h2.title{margin:8px 0 4px;font-size:19px;word-break:break-word}
h3{font-size:13px;text-transform:uppercase;color:#666;margin:18px 0 6px;border-bottom:1px solid #eee;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px}
table.props th{text-align:left;width:180px;color:#555;vertical-align:top;padding:5px 8px;border-bottom:1px solid #f0f0f2}
table.props th[title]{cursor:help}.pinfo{opacity:.55;font-size:11px}
table.props td{padding:5px 8px;border-bottom:1px solid #f0f0f2;word-break:break-word}
table.rels td{padding:5px 8px;border-bottom:1px solid #f0f0f2}
.dir{color:#999;font-size:11px;text-transform:uppercase}.rel{color:#888}
.copy{opacity:0;border:none;background:none;cursor:pointer;font-size:12px;color:#889;
  margin-left:6px;padding:0 2px;vertical-align:middle;transition:opacity .12s}
td:hover .copy{opacity:.5}.copy:hover{opacity:1}.copy.ok{opacity:1;color:#1f9d55}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f7f9;padding:8px;border-radius:4px;font-size:12px;margin:0}
.jt{margin:0}.jt>summary{cursor:pointer;color:#2657c7;font-size:12px;font-family:ui-monospace,monospace}
.jbody{border-left:1px solid #e6e6ea;margin-left:5px;padding-left:10px}
.jrow{padding:1px 0;font-family:ui-monospace,monospace;font-size:12px;word-break:break-word}
.jkey{color:#8a4fbf}.jstr{color:#1f9d55}.jnum{color:#2657c7}.jbool{color:#c2681f}.jnull{color:#999}
.jpunct{color:#999;font-family:ui-monospace,monospace;font-size:12px}
.empty{color:#999;font-style:italic}
.btn{background:#eef1fb;color:#2657c7;border:none;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:13px}
.md{background:#fffdf5;border:1px solid #f0ead0;padding:2px 14px;border-radius:6px}
.md h1,.md h2,.md h3{border:0;color:#1a1a1a;margin:12px 0 6px}
.md h1{font-size:20px}.md h2{font-size:17px}.md h3{font-size:15px;text-transform:none}
.md pre{background:#f2efe4}.md code{background:#f2efe4;padding:1px 4px;border-radius:3px;font-size:12px}
.md table{margin:8px 0}.md th,.md td{border:1px solid #e6ddc4;padding:4px 8px}
.md img{max-width:100%}.md blockquote{border-left:3px solid #d8ceac;margin:8px 0;padding-left:12px;color:#666}
ul.plain{margin:6px 0;padding-left:18px}ul.plain li{margin:2px 0}
"""

_HISTORY_JS = """
(function(){
  var KEY='rag_view_history';
  var LETTER={relation:'R',chunk:'C',entity:'E',document:'D',search:'S',home:'H'};
  function kindOf(url){
    if(url.indexOf('/relation')===0)return 'relation';
    if(url.indexOf('/search')===0)return 'search';
    if(url.indexOf('/item')===0){
      var m=url.match(/[?&]kind=([^&]+)/);
      return m?decodeURIComponent(m[1]):'entity';
    }
    return 'home';
  }
  function load(){try{return JSON.parse(localStorage.getItem(KEY))||[]}catch(e){return[]}}
  function save(h){localStorage.setItem(KEY,JSON.stringify(h.slice(-50)))}
  var title=document.title.replace(/^rag · /,'');
  var url=location.pathname+location.search;
  var h=load();
  if(!h.length||h[h.length-1].url!==url){h.push({title:title,url:url,kind:kindOf(url)});save(h)}
  var ol=document.getElementById('history');
  if(ol){h.slice().reverse().forEach(function(e){
    var kind=e.kind||kindOf(e.url);
    var li=document.createElement('li');
    var b=document.createElement('span');
    b.className='hb '+kind; b.textContent=LETTER[kind]||'?';
    var a=document.createElement('a');a.href=e.url;a.textContent=e.title;
    li.appendChild(b);li.appendChild(a);ol.appendChild(li);
  });}
  var clr=document.getElementById('clearhist');
  if(clr)clr.onclick=function(){localStorage.removeItem(KEY);location.reload()};
})();
"""

_COPY_JS = """
document.addEventListener('click',function(e){
  var b=e.target.closest('.copy'); if(!b)return;
  var v=b.getAttribute('data-copy')||'';
  var done=function(){var o=b.textContent;b.textContent='✓';b.classList.add('ok');
    setTimeout(function(){b.textContent=o;b.classList.remove('ok')},900)};
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(v).then(done,function(){});
  }else{
    var t=document.createElement('textarea');t.value=v;document.body.appendChild(t);
    t.select();try{document.execCommand('copy')}catch(_){}document.body.removeChild(t);done();
  }
});
"""


def _search_form(query: str = "") -> str:
    return (
        "<form action=/search method=get class=search>"
        f"<input name=q placeholder='Search or paste ID' value=\"{html.escape(query)}\">"
        "<label class=sem title=\"Off: literal text match over the graph (entity names, "
        "relation labels, chunk text, document paths). On: also embed your query with the "
        "model and run a vector similarity search in Qdrant — finds passages that mean the "
        "same thing even when they share no words. Costs one embedding call, so it is slower "
        'and needs the endpoint reachable.">'
        "<input type=checkbox name=semantic value=1> semantic <span class=info>ⓘ</span>"
        "</label>"
        "<button class=btn>Search</button></form>"
    )


def _page(title: str, body: str, query: str = "") -> str:
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        f"<title>rag · {html.escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<header><h1><a href=/>rag viewer</a></h1>"
        "<button class=btn onclick='history.back()'>← Back</button>"
        f"{_search_form(query)}"
        "</header><div class=layout><main>"
        f"{body}"
        "</main><aside><h2>History</h2><ol id=history></ol>"
        "<button class=btn id=clearhist style='margin-top:10px'>clear</button>"
        "</aside></div>"
        f"<script>{_HISTORY_JS}{_COPY_JS}</script></body></html>"
    )


@app.route("/")
def index():
    graph = _graph()
    nodes = _nodes(graph)
    entities = sorted(
        (nid for nid, n in nodes.items() if _is_entity(graph, nid, n)),
        key=str.lower,
    )
    docs = sorted(_doc_ids())
    n_rel = len(_relations(graph))
    ent_list = "".join(f"<li>{_link(graph, e)}</li>" for e in entities[:500])
    doc_list = "".join(
        f'<li><a href="{html.escape(_url("document", d))}">{html.escape(d)}</a></li>'
        for d in docs
    )
    n_embed = len(_DOCS.state("embed_state"))
    n_graph = len(_DOCS.state("graph_state"))
    body = (
        "<div class=card><h2 class=title>Stores</h2>"
        f"<p>{len(entities)} entities · {n_rel} relations · {len(docs)} documents<br>"
        f"({n_embed} embedded, {n_graph} graph-extracted)</p></div>"
        f"<div class=card><h3>Documents</h3><ul class=plain>{doc_list or '<li class=empty>none</li>'}</ul></div>"
        f"<div class=card><h3>Entities</h3><ul class=plain>{ent_list or '<li class=empty>none</li>'}</ul></div>"
    )
    return _page("home", body)


def _item_response(graph, item_id: str, kind: str | None = None):
    kind = kind or _kind_of(graph, item_id)
    if kind == "relation":
        return _relation_response(graph, item_id)
    if kind == "document":
        return _page(_display(graph, item_id), _document_body(graph, item_id))
    node = _nodes(graph).get(item_id)
    if node is None:
        return _page("not found", f"<div class=card><p class=empty>No item "
                     f"'{html.escape(item_id)}'.</p></div>"), 404
    if kind == "chunk":
        return _page(_display(graph, item_id), _chunk_body(graph, item_id, node))
    return _page(_display(graph, item_id), _entity_body(graph, item_id, node))


@app.route("/item")
def item():
    graph = _graph()
    return _item_response(graph, request.args.get("id", ""), request.args.get("kind"))


def _entity_body(graph, item_id, node) -> str:
    rels = [r for r in _relations(graph) if item_id in (r.source_id, r.target_id)]
    rel_rows = "".join(_relation_row(graph, r, item_id) for r in rels) or (
        "<tr><td colspan=4 class=empty>no relations</td></tr>"
    )
    return (
        f"<div class=card><span class=\"kind entity\">entity</span>"
        f"<h2 class=title>{html.escape(getattr(node,'name',item_id))}</h2>"
        f"<h3>Properties</h3>{_prop_table(graph, getattr(node,'properties',{}) or {}, kind='entity')}"
        f"<h3>Relations ({len(rels)})</h3><table class=rels>{rel_rows}</table></div>"
    )


def _chunk_body(graph, item_id, node) -> str:
    text = getattr(node, "text", "") or ""
    props = dict(getattr(node, "properties", {}) or {})
    entities = [
        nid for nid, n in _nodes(graph).items()
        if _is_entity(graph, nid, n)
        and (getattr(n, "properties", {}) or {}).get("triplet_source_id") == item_id
    ]
    rels = [r for r in _relations(graph) if item_id in (r.source_id, r.target_id)]
    rel_rows = "".join(_relation_row(graph, r, item_id) for r in rels) or (
        "<tr><td colspan=4 class=empty>no relations</td></tr>"
    )
    ent_list = "".join(f"<li>{_link(graph, e)}</li>" for e in entities) or "<li class=empty>none</li>"
    return (
        f"<div class=card><span class=\"kind chunk\">chunk</span>"
        f"<h2 class=title>chunk {html.escape(item_id[:8])}</h2>"
        f"<h3>Text</h3><div class=md>{_render_md(text)}</div>"
        f"<h3>Properties</h3>{_prop_table(graph, props, kind='chunk')}"
        f"<h3>Entities extracted here ({len(entities)})</h3><ul class=plain>{ent_list}</ul>"
        f"<h3>Relations ({len(rels)})</h3><table class=rels>{rel_rows}</table></div>"
    )


def _document_body(graph, doc_id) -> str:
    """Document view from the canonical Document Store: metadata + full text."""
    sdoc = _DOCS.get(doc_id) if _DOCS is not None else None
    if sdoc is None:
        return "<div class=card><p class=empty>Document not found.</p></div>"
    meta = {k: v for k, v in sdoc.metadata.items() if k != "_node_content"}
    # Documents are referred to by doc_id (shown as the title). For the external
    # open-link, translate the raw ref (doc_id minus the "extractor:" prefix).
    raw = doc_id.split(":", 1)[1] if ":" in doc_id else doc_id
    url = _translate_path(raw)
    title = (f'<a href="{html.escape(url, quote=True)}">{html.escape(doc_id)}</a>'
             if url else html.escape(doc_id))
    return (
        f"<div class=card><span class=\"kind document\">document</span>"
        f"<h2 class=title>{title}</h2>"
        f"<h3>Metadata</h3>{_prop_table(graph, meta, linkify=False, kind='document')}"
        f"<h3>Text</h3><div class=md>{_render_md(sdoc.text)}</div></div>"
    )


# --- search ---------------------------------------------------------------

def _score(text: str, ql: str) -> int:
    tl = text.lower()
    if tl == ql:
        return 100
    if tl.startswith(ql):
        return 80
    if ql in tl:
        return 50
    return 0


def _snippet(text: str, q: str, width: int = 120) -> str:
    flat = " ".join(text.split())
    i = flat.lower().find(q.lower())
    if i < 0:
        return flat[:width]
    start = max(0, i - 30)
    return ("…" if start else "") + flat[start:start + width] + "…"


def _text_search(graph, q: str, limit: int = 100):
    """Rank graph items by a substring match. Returns (score, kind, ref, label)."""
    ql = q.lower()
    hits = []
    for nid, node in _nodes(graph).items():
        if _is_entity(graph, nid, node):
            name = getattr(node, "name", nid) or nid
            s = _score(str(name), ql)
            if s:
                hits.append((s + 5, "entity", nid, str(name)))  # weight entities
        elif getattr(node, "label", "") == "text_chunk":
            text = getattr(node, "text", "") or ""
            if ql in text.lower():
                hits.append((35, "chunk", nid, _snippet(text, q)))
        # id prefix match (paste a shortened id like 3b1c8a10)
        if nid.lower().startswith(ql) and nid.lower() != ql:
            hits.append((90, _kind_of(graph, nid), nid, "id: " + nid))
    for d in _doc_ids():
        s = _score(d, ql)
        if s:
            hits.append((s - 10, "document", d, d))
    for r in _relations(graph):
        if ql in f"{r.source_id} {r.label} {r.target_id}".lower():
            label = f"{r.source_id} -[{r.label}]-> {r.target_id}"
            hits.append((45, "relation", r, label))
    hits.sort(key=lambda h: h[0], reverse=True)
    return hits[:limit]


def _vector_search(q: str, n: int = 10):
    """Semantic search over Qdrant. Returns (points, error). Best-effort: any
    failure (endpoint down/auth/dim) yields ([], message) instead of raising."""
    try:
        runtime.configure(_CFG)
        emb = runtime.Settings.embed_model.get_query_embedding(q)
    except Exception as exc:  # endpoint/auth/etc.
        return [], f"embedding failed: {exc}"
    try:
        client = stores.open_qdrant_client(_CFG)
    except Exception as exc:
        # Embedded Qdrant (`path:`) allows one process at a time — if ingest/chat
        # holds the lock, opening here raises. Surface it as a note, not a 500.
        return [], f"Qdrant unavailable: {exc}"
    try:
        coll = _CFG.qdrant.collection
        names = {c.name for c in client.get_collections().collections}
        if coll not in names:
            return [], "collection does not exist yet"
        res = client.query_points(coll, query=emb, limit=n, with_payload=True).points
        return res, None
    except Exception as exc:
        return [], f"query failed: {exc}"
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _hit_row(graph, score, kind, ref, label) -> str:
    if kind == "relation":
        link = f'<a href="{html.escape(_rel_url(ref))}">{html.escape(label)}</a>'
    else:
        link = _link(graph, ref, _display(graph, ref) if ref in _nodes(graph) else label)
    return (
        f"<div class=hit><span class=score>{score}</span>"
        f"<span class=badge>{html.escape(kind)}</span>"
        f"<span>{link} <span class=snip>{html.escape(label) if kind=='chunk' else ''}</span></span></div>"
    )


@app.route("/search")
def search():
    graph = _graph()
    q = request.args.get("q", "").strip()
    semantic = request.args.get("semantic")
    if not q:
        return _page("search", "<div class=card><p class=empty>Enter a query or ID.</p></div>")

    # Exact id / document -> jump straight to the item.
    if q in _doc_ids():
        return _item_response(graph, q, "document")
    if q in _nodes(graph):
        return _item_response(graph, q)
    # Unique shortened-id prefix -> jump to it.
    prefix = [nid for nid in _nodes(graph) if nid.lower().startswith(q.lower())]
    if len(q) >= 6 and len(prefix) == 1:
        return _item_response(graph, prefix[0])

    rows = "".join(_hit_row(graph, *h) for h in _text_search(graph, q))
    text_section = (
        f"<div class=card><h3>Graph matches</h3>{rows or '<p class=empty>no matches</p>'}</div>"
    )

    sem_section = ""
    if semantic:
        points, err = _vector_search(q)
        if err:
            sem_section = f"<div class=card><h3>Semantic matches</h3><p class=empty>unavailable — {html.escape(err)}</p></div>"
        else:
            srows = []
            for p in points:
                payload = p.payload or {}
                text = payload.get("text") or _text_from_node(payload) or ""
                pid = str(p.id)
                link = _link(graph, pid, _display(graph, pid) if pid in _nodes(graph) else pid[:8])
                srows.append(
                    f"<div class=hit><span class=score>{p.score:.2f}</span>"
                    f"<span class=badge>vector</span>"
                    f"<span>{link} <span class=snip>{html.escape(_snippet(text, q))}</span></span></div>"
                )
            sem_section = (
                f"<div class=card><h3>Semantic matches (vector)</h3>"
                f"{''.join(srows) or '<p class=empty>no matches</p>'}</div>"
            )

    header = f"<div class=card><h2 class=title>Search: {html.escape(q)}</h2></div>"
    return _page(f"search: {q}", header + text_section + sem_section, query=q)


def _text_from_node(payload: dict):
    raw = payload.get("_node_content")
    if not raw:
        return None
    try:
        return json.loads(raw).get("text")
    except Exception:
        return None


def _relation_props(source: str, label: str, target: str) -> dict:
    """Fetch a relation's stored properties (incl provenance triplet_source_id).

    Kùzu's get_triplets drops relation properties, so query the edge directly.
    """
    _ensure_stores()
    try:
        rows = _GRAPH.structured_query(
            "MATCH (e)-[r]->(t) WHERE e.id = $s AND t.id = $t AND r.label = $l "
            "RETURN r LIMIT 1",
            param_map={"s": source, "t": target, "l": label},
        )
    except Exception:
        return {}
    if not rows:
        return {}
    rel = rows[0].get("r", {}) or {}
    skip = {"_src", "_dst", "_id", "_label", "label"}
    return {k: v for k, v in rel.items() if k not in skip}


def _relation_response(graph, rid: str):
    match = _rel_from_id(graph, rid)
    if match is None:
        return _page("not found",
                     "<div class=card><p class=empty>Relation not found.</p></div>"), 404
    source, label, target = match.source_id, match.label, match.target_id
    props = _relation_props(source, label, target)
    body = (
        f"<div class=card><span class=\"kind relation\">relation</span>"
        f"<h2 class=title>{_link(graph, source)} "
        f"-[{html.escape(label)}]-&gt; {_link(graph, target)}</h2>"
        f"<h3>Source</h3><p>{_link(graph, source)}</p>"
        f"<h3>Target</h3><p>{_link(graph, target)}</p>"
        f"<h3>Properties</h3>{_prop_table(graph, props, kind='relation')}</div>"
    )
    return _page(f"{label}", body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.viewer", description="Browse the stores in a local web app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="auto-restart on source edits (watches viewer.py + properties.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global _CFG
    args = build_parser().parse_args(argv)
    _CFG = config.load(".")
    runtime.configure(_CFG)  # embed model for semantic search
    # Stores are opened lazily on first request (in the worker), so the reloader
    # parent never grabs the Kùzu single-writer lock. The idle watcher releases
    # them after inactivity so pipelines can run without stopping the viewer.
    _start_idle_watcher()
    print(f"rag viewer on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    # Reloader without the Werkzeug debugger (no in-browser RCE console; keeps our
    # custom error page). extra_files picks up the non-.py descriptions file.
    props = str(Path(__file__).parent.parent / "properties.yaml")
    try:
        app.run(
            host=args.host,
            port=args.port,
            use_reloader=args.reload,
            use_debugger=False,
            # Single-threaded: SQLite / Kùzu connections are opened on this thread
            # and requests must run on it too. Fine for a local single-user viewer.
            threaded=False,
            extra_files=[props] if args.reload else None,
        )
    finally:
        # Best-effort: on reload the restart runs on a different thread than the
        # one that opened SQLite/Kùzu, so closing may raise — the process is
        # exiting anyway, so swallow it.
        try:
            if _GRAPH_DB is not None:
                stores.close_graph_db(_GRAPH_DB)
        except Exception:
            pass
        try:
            if _DOCS is not None:
                _DOCS.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
