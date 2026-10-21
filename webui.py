"""qillqa web: navegar y buscar páginas (FastAPI). Lo arranca el MCP (start_server).

Índice semántico en var/index-<modelo>.db, sincronizado por mtime antes de cada
búsqueda y tras cada escritura del MCP (POST /api/sync). El modelo corre en un
proceso hijo (embedder.py) que se cierra tras QILLQA_MODEL_IDLE_MIN minutos sin
uso, así que el servidor queda en ~76 MB y sin torch.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import html
import multiprocessing
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import markdown
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

import embedder
from server import MODEL, PAGES, TOKEN, VAR, _index, _rel, _safe, _title, read_page, tree

IDLE_MIN = float(os.environ.get("QILLQA_MODEL_IDLE_MIN", "15"))
# Calibrado con Qwen3-Embedding-0.6B: aciertos 0.42-0.60, ruido <= 0.21
MIN_SCORE = float(os.environ.get("QILLQA_MIN_SCORE", "0.3"))
DB = VAR / f"index-{hashlib.sha1(MODEL.encode()).hexdigest()[:8]}.db"  # otro modelo = otro índice
COOKIE = "qillqa"
SESSION = hmac.new(TOKEN.encode(), b"qillqa-session", hashlib.sha256).hexdigest()

# ---------------------------------------------------------------- embeddings

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_last_use = 0.0


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    """Vectores normalizados (float32). El modelo vive en un proceso hijo."""
    global _pool, _last_use
    with _pool_lock:  # también impide que el reaper cierre el hijo a mitad de una llamada
        if _pool is None:
            _pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        v = _pool.submit(embedder.encode, texts, query).result()
        _last_use = time.time()
    return v


def _reaper() -> None:
    global _pool
    while True:
        time.sleep(30)
        with _pool_lock:
            if _pool is not None and time.time() - _last_use > IDLE_MIN * 60:
                _pool.shutdown()  # el hijo termina: RAM y VRAM vuelven al sistema
                _pool = None


# ---------------------------------------------------------------- índice

_BLOCKS = {"title", "p", "div", "li", "br", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
           "section", "article", "header", "footer", "pre", "blockquote", "ul", "ol", "table"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self._skip -= tag in ("script", "style") and self._skip > 0
        if tag in _BLOCKS:
            self.out.append("\n\n")

    def handle_data(self, data):
        if not self._skip:
            self.out.append(data)


def page_text(idx: Path) -> str:
    text = idx.read_text(errors="replace")
    if idx.suffix == ".html":
        p = _Text()
        p.feed(text)
        text = re.sub(r"[ \t]+", " ", "".join(p.out))
    return text.strip()


def chunks(title: str, text: str, size: int = 1500) -> list[str]:
    out, cur = [], ""
    for part in re.split(r"\n(?=#{1,6} )|\n\s*\n", text):
        part = part.strip()
        if not part:
            continue
        if cur and len(cur) + len(part) + 2 > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{part}" if cur else part
        while len(cur) > size:
            out.append(cur[:size])
            cur = cur[size:]
    if cur:
        out.append(cur)
    return [f"{title}\n\n{c}" for c in out] or [title]


_sync_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    VAR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("create table if not exists chunks"
                "(path text, mtime real, i int, text text, vec blob)")
    return con


def _pages() -> dict[str, Path]:
    found = {}
    for d in [PAGES, *PAGES.rglob("*")]:
        if not d.is_dir() or any(p.startswith(".") for p in d.relative_to(PAGES).parts):
            continue
        idx = _index(d)
        if idx and PAGES in idx.resolve().parents:  # sin symlinks hacia fuera
            found[_rel(d)] = idx
    return found


def sync() -> int:
    """Re-embebe las páginas nuevas o cambiadas y olvida las borradas."""
    changed = 0
    with _sync_lock, contextlib.closing(_db()) as con:
        known = dict(con.execute("select path, max(mtime) from chunks group by path"))
        pages = _pages()
        for gone in set(known) - set(pages):
            con.execute("delete from chunks where path=?", (gone,))
        for rel, idx in pages.items():
            mtime = idx.stat().st_mtime
            if known.get(rel) == mtime:
                continue
            pieces = chunks(_title(idx.parent), page_text(idx))
            vecs = embed(pieces)
            con.execute("delete from chunks where path=?", (rel,))
            con.executemany("insert into chunks values (?,?,?,?,?)",
                            [(rel, mtime, i, t, v.tobytes())
                             for i, (t, v) in enumerate(zip(pieces, vecs))])
            changed += 1
        con.commit()
    return changed


def search(q: str, limit: int = 10) -> list[dict]:
    sync()
    with contextlib.closing(_db()) as con:
        rows = con.execute("select path, text, vec from chunks").fetchall()
    if not rows or not q.strip():
        return []
    # ponytail: fuerza bruta en numpy; pasar a sqlite-vec si supera ~100k chunks
    m = np.frombuffer(b"".join(r[2] for r in rows), np.float32).reshape(len(rows), -1)
    scores = m @ embed([q], query=True)[0]
    best: dict[str, dict] = {}
    for k in np.argsort(-scores):
        if scores[k] < MIN_SCORE:
            break
        path, text = rows[k][0], rows[k][1]
        if path not in best:
            title, _, body = text.partition("\n\n")
            best[path] = {"path": path, "title": title, "score": round(float(scores[k]), 4),
                          "snippet": body[:300]}
            if len(best) >= limit:
                break
    return list(best.values())


# ---------------------------------------------------------------- app

@contextlib.asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_reaper, daemon=True).start()
    yield


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def _eq(a: str, b: str) -> bool:
    """compare_digest en bytes: con str y caracteres no ASCII lanza TypeError."""
    return hmac.compare_digest(a.encode(), b.encode())


def _authed(req: Request) -> bool:
    auth = req.headers.get("authorization", "")
    if TOKEN and auth.startswith("Bearer ") and _eq(auth[7:].strip(), TOKEN):
        return True
    return bool(TOKEN) and _eq(req.cookies.get(COOKIE, ""), SESSION)


@app.middleware("http")
async def gate(req: Request, call_next):
    if req.url.path in ("/health", "/login") or _authed(req):
        return await call_next(req)
    if req.url.path.startswith(("/api/", "/view")):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/health")
def health():
    return {"app": "qillqa", "ok": True}


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    msg = '<p class="err">Token incorrecto</p>' if error else ""
    return _shell("Entrar", f"""<form method="post" action="/login" class="login">
<h1>qillqa</h1>{msg}<input type="password" name="token" placeholder="Token" autofocus required>
<button>Entrar</button></form>""")


@app.post("/login")
async def login(req: Request):
    token = urllib.parse.parse_qs((await req.body()).decode()).get("token", [""])[0]
    if not TOKEN or not _eq(token.strip(), TOKEN):
        return RedirectResponse("/login?error=1", status_code=303)
    r = RedirectResponse("/", status_code=303)
    r.set_cookie(COOKIE, SESSION, httponly=True, samesite="strict", max_age=90 * 86400)
    return r


@app.get("/logout")
def logout():
    r = RedirectResponse("/login", status_code=303)
    r.delete_cookie(COOKIE)
    return r


def _bad(e: Exception):
    raise HTTPException(404, str(e))


@app.get("/api/tree")
def api_tree():
    return tree()


@app.get("/api/page")
def api_page(path: str = ""):
    try:
        p = read_page(path)
    except ValueError as e:
        _bad(e)
    p.pop("content")
    return p


@app.post("/api/sync")
def api_sync():
    """Indexa en segundo plano; el MCP lo llama tras crear/editar/mover/borrar."""
    threading.Thread(target=sync, daemon=True).start()
    return {"ok": True}


@app.get("/api/search")
def api_search(q: str = "", limit: int = 10):
    return search(q, max(1, min(limit, 50)))


@app.get("/view")
def view(path: str = ""):
    """La página tal cual, siempre aislada: CSP sandbox = origen opaco, sin cookie ni API."""
    try:
        idx = _index(_safe(path))
    except ValueError as e:
        _bad(e)
    if not idx:
        _bad(ValueError(f"no hay página en {path!r}"))
    text = idx.read_text(errors="replace")
    if idx.suffix == ".html":
        sandbox = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals"
        body = text
    else:
        sandbox = "sandbox allow-popups allow-popups-to-escape-sandbox"
        body = _shell(_title(idx.parent), '<base target="_blank"><article>'
                      + markdown.markdown(text, extensions=["fenced_code", "tables", "toc", "sane_lists"])
                      + "</article>")
    return Response(body, media_type="text/html; charset=utf-8",
                    headers={"Content-Security-Policy": sandbox, "Cache-Control": "no-store"})


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1d1a;--mute:#6b665e;--line:#e4e0d8;--card:#fff;--acc:#a8462b;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root{--bg:#161513;--fg:#ece8e1;--mute:#a19b91;--line:#2f2c28;--card:#1e1c1a;--acc:#e0805f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}
a{color:var(--acc)}code,pre{font-family:ui-monospace,monospace;font-size:.9em}
pre{background:var(--card);border:1px solid var(--line);padding:12px;overflow:auto;border-radius:6px}
table{border-collapse:collapse}td,th{border:1px solid var(--line);padding:4px 8px}
article{max-width:780px;margin:0 auto;padding:24px 16px}img{max-width:100%}
.login{max-width:320px;margin:20vh auto;display:grid;gap:12px;padding:0 16px}
input,button{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer;background:var(--acc);color:#fff;border:0}.err{color:var(--acc);margin:0}
"""


def _shell(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>")


INDEX = """
<header><a href="#" class="brand">qillqa</a>
<input id="q" type="search" placeholder="Buscar por significado…" autocomplete="off">
<a href="/logout" class="out">salir</a></header>
<main><aside><button id="toggle">Páginas</button><nav id="tree"></nav></aside>
<section><div id="results" hidden></div><div id="crumb"></div>
<iframe id="page" sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals" title="página"></iframe>
<p id="empty">Elige una página o busca algo.</p></section></main>
<style>
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
.brand{font-weight:700;text-decoration:none;color:var(--fg)}#q{flex:1;min-width:0}.out{color:var(--mute);font-size:.9em}
main{display:grid;grid-template-columns:280px 1fr;height:calc(100vh - 59px)}
aside{border-right:1px solid var(--line);overflow:auto;padding:8px}#toggle{display:none}
nav ul{list-style:none;margin:0;padding-left:14px}nav>ul{padding-left:0}
nav a{display:block;padding:3px 6px;border-radius:4px;color:var(--fg);text-decoration:none;overflow-wrap:anywhere}
nav a.sec{color:var(--mute)}nav a.on{background:var(--line)}
section{display:flex;flex-direction:column;min-width:0}iframe{flex:1;border:0;width:100%;background:var(--bg)}
#crumb{padding:6px 16px;color:var(--mute);font-size:.85em;overflow-wrap:anywhere}#empty{padding:24px 16px;color:var(--mute)}
#results{padding:8px 16px;border-bottom:1px solid var(--line);max-height:45vh;overflow:auto}
.hit{display:block;padding:8px 0;border-bottom:1px solid var(--line);color:var(--fg);text-decoration:none}
.hit b{color:var(--acc)}.hit small{color:var(--mute);display:block}
@media (max-width:760px){main{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid var(--line)}
#toggle{display:block;width:100%;background:var(--card);color:var(--fg);border:1px solid var(--line)}
nav{display:none}nav.open{display:block}iframe{height:75vh;flex:none}}
</style>
<script>
const $ = s => document.querySelector(s);
function mk(tag, props, ...kids){const e=Object.assign(document.createElement(tag),props);e.append(...kids);return e}
async function api(u){const r=await fetch(u);if(r.status===401)location='/login';return r.json()}
function renderTree(nodes){
  const ul=mk('ul');
  for(const n of nodes){
    const a=mk('a',{href:'#'+n.path,className:n.page?'':'sec',textContent:n.title});a.dataset.path=n.path;
    const li=mk('li',{},a);if(n.children.length)li.append(renderTree(n.children));ul.append(li);
  }
  return ul;
}
async function loadTree(){$('#tree').replaceChildren(renderTree(await api('/api/tree')));mark()}
function mark(){const p=decodeURIComponent(location.hash.slice(1));
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.dataset.path===p))}
async function show(){
  const p=decodeURIComponent(location.hash.slice(1));mark();
  if(!p){$('#page').hidden=true;$('#empty').hidden=false;$('#crumb').textContent='';return}
  $('#crumb').textContent=p.split('/').join(' / ');$('#empty').hidden=true;$('#page').hidden=false;
  $('#page').src='/view?path='+encodeURIComponent(p);$('nav').classList.remove('open');
}
let t;
$('#q').addEventListener('input',()=>{clearTimeout(t);t=setTimeout(find,400)});
async function find(){
  const q=$('#q').value.trim(),box=$('#results');
  if(!q){box.hidden=true;return}
  box.hidden=false;box.replaceChildren(mk('small',{textContent:'Buscando… (la primera vez carga el modelo)'}));
  const hits=await api('/api/search?q='+encodeURIComponent(q));
  box.replaceChildren(...(hits.length?hits.map(h=>mk('a',{className:'hit',href:'#'+h.path},
    mk('b',{textContent:h.title}),mk('small',{textContent:h.path+' · '+h.score}),
    mk('span',{textContent:h.snippet}))):[mk('small',{textContent:'Sin resultados'})]));
}
$('#toggle').onclick=()=>$('nav').classList.toggle('open');
addEventListener('hashchange',show);loadTree().then(show);
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _shell("qillqa", INDEX)
