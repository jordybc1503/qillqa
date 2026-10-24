"""qillqa web: browse and search pages (FastAPI). Started by the MCP (start_server).

Semantic index in var/index-<model>.db, synced by mtime before every search and
after every MCP write (POST /api/sync). The model runs in a child process
(embedder.py) that exits after QILLQA_MODEL_IDLE_MIN idle minutes, so the server
itself stays at ~76 MB with no torch loaded.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import html
import multiprocessing
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import markdown
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

import embedder
import server
from server import MODEL, PAGES, TOKEN, VAR, _index, _rel, _safe, _title, read_page, tree

IDLE_MIN = float(os.environ.get("QILLQA_MODEL_IDLE_MIN", "15"))
# Calibrated on Qwen3-Embedding-0.6B: hits 0.41-0.65, noise <= 0.21
MIN_SCORE = float(os.environ.get("QILLQA_MIN_SCORE", "0.3"))
DB = VAR / f"index-{hashlib.sha1(MODEL.encode()).hexdigest()[:8]}.db"  # another model = another index
COOKIE = "qillqa"
SESSION = hmac.new(TOKEN.encode(), b"qillqa-session", hashlib.sha256).hexdigest()

# ---------------------------------------------------------------- embeddings

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_last_use = 0.0


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    """Normalized vectors (float32). The model lives in a child process."""
    global _pool, _last_use
    with _pool_lock:  # also keeps the reaper from closing the child mid-call
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
                _pool.shutdown()  # the child exits: RAM and VRAM go back to the system
                _pool = None


# ---------------------------------------------------------------- index

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
        if idx and PAGES in idx.resolve().parents:  # no symlinks escaping the notebook
            found[_rel(d)] = idx
    return found


def sync() -> int:
    """Re-embed new or changed pages and forget deleted ones."""
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


def _plain(md: str) -> str:
    """Snippet text without Markdown/HTML syntax."""
    md = re.sub(r"```\w*|<[^>]+>|!\[([^\]]*)\]\([^)]*\)|\[([^\]]*)\]\([^)]*\)", r"\1\2", md)
    md = re.sub(r"^\s*(#+|[-*>](?=\s)|\|?-{3,}.*)\s*", "", md, flags=re.M)
    return re.sub(r"\s+", " ", md.replace("**", "").replace("`", "").replace("|", " ")).strip()


def search(q: str, limit: int = 10, path: str = "") -> list[dict]:
    """`path` limits the search to that folder (a project) and everything under it."""
    root = _safe(path)
    if not root.is_dir():
        raise ValueError(f"does not exist: {path!r}")
    root = _rel(root)
    sync()
    with contextlib.closing(_db()) as con:
        rows = con.execute("select path, text, vec from chunks").fetchall()
    if root:
        rows = [r for r in rows if r[0] == root or r[0].startswith(root + "/")]
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
                          "snippet": _plain(body)[:300]}
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
    """compare_digest on bytes: on str with non-ASCII chars it raises TypeError."""
    return hmac.compare_digest(a.encode(), b.encode())


def _authed(req: Request) -> bool:
    auth = req.headers.get("authorization", "")
    if TOKEN and auth.startswith("Bearer ") and _eq(auth[7:].strip(), TOKEN):
        return True
    return bool(TOKEN) and _eq(req.cookies.get(COOKIE, ""), SESSION)


@app.middleware("http")
async def gate(req: Request, call_next):
    if req.url.path in ("/health", "/login") or req.url.path.startswith("/f/") or _authed(req):
        return await call_next(req)
    if req.url.path.startswith(("/api/", "/view")):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/health")
def health():
    return {"app": "qillqa", "ok": True}


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    msg = '<p class="err">Wrong token</p>' if error else ""
    return _shell("Sign in", f"""<form method="post" action="/login" class="login">
<h1>qillqa</h1>{msg}<input type="password" name="token" placeholder="Token" autofocus required>
<button>Sign in</button></form>""")


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
    """Index in the background; the MCP calls it after create/update/move/delete."""
    threading.Thread(target=sync, daemon=True).start()
    return {"ok": True}


@app.post("/api/page/edit")
async def api_page_edit(req: Request):
    """Checklist edit from the shell: {path, op: check|move, i, to?, after?, cap}. `cap` comes
    from one render of that page; 409 if the file changed since, so a stale page never overwrites."""
    if not req.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "json only")
    d = await req.json()
    try:
        page = _safe(str(d.get("path", "")))
        idx = _index(page) if page != PAGES else None
        if not idx or idx.suffix != ".md":
            raise ValueError("not a markdown page")
        with _edit_lock:
            granted = _caps.get(str(d.get("cap", "")))
        if not granted or granted[0] != _rel(page):
            raise HTTPException(403, "this page did not render that checklist")
        with server.page_lock():
            text = _read_raw(idx)
            if granted[1] != _version(text):
                return JSONResponse({"error": "the page changed, reloaded"}, status_code=409)
            new = edit_checklist(text, str(d.get("op")), int(d.get("i", -1)),
                                 None if d.get("to") is None else int(d["to"]), bool(d.get("after")))
            fd, tmp = tempfile.mkstemp(dir=idx.parent, prefix=".edit-", suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(new.encode())
            os.replace(tmp, idx)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    threading.Thread(target=sync, daemon=True).start()
    return {"ok": True}


@app.get("/api/search")
def api_search(q: str = "", limit: int = 10, path: str = ""):
    try:
        return search(q, max(1, min(limit, 50)), path)
    except ValueError as e:
        _bad(e)


# ---------------------------------------------------------------- archivos
# Every page is served under /f/<sig>/<folder>/ so relative links in the md or
# html (photos, videos, .css, .js, PDFs) work without a cookie, because the page
# runs sandboxed (opaque origin). A signature only opens that folder and its
# subpages.
# ponytail: signatures never expire; add expiry if this is ever exposed to the internet.

def _sig(rel: str) -> str:
    return hmac.new(TOKEN.encode(), b"f:" + rel.encode(), hashlib.sha256).hexdigest()[:32]


def signed_url(rel: str) -> str:
    return f"/f/{_sig(rel)}/" + "".join(urllib.parse.quote(seg) + "/" for seg in rel.split("/"))


_HTML_SANDBOX = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals"
_MD_SANDBOX = "sandbox allow-popups allow-popups-to-escape-sandbox"
_MD_LIVE_SANDBOX = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox"
_SAFE_HEADERS = {"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"}
_VIDEO = r"mp4|webm|mov|m4v"
_AUDIO = r"mp3|wav|m4a|ogg|flac|opus"


# ---------------------------------------------------------------- checklists
# A markdown page with ☐/☑/◐ in a table row, or at the start of a list item, is a
# checklist: click a box to flip it, drag a row by its ⠿ grip to another place or
# another table. The md file stays the source of truth: the sandboxed page posts
# the change to the shell (which has the session), the shell calls
# POST /api/page/edit, the page reloads at the same scroll.
# Which boxes are clickable is read from markdown's own html (every box in the
# source is numbered first), so a ☐ in a sentence, a link or a code block stays text.
# Each render gets a capability (its CSP nonce) bound to that page and that version
# of the file: a stale page or another document in the iframe cannot edit.

BOXES = "☐☑◐"
_BOX = re.compile(f"[{BOXES}]")
_PUA = re.compile("[\ue000-\ue002]")
_MARK = re.compile(f"\ue000(\\d+)\ue001([{BOXES}])\ue002")
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_caps: dict[str, tuple[str, str]] = {}  # capability -> (page, version); newest last
_edit_lock = threading.Lock()


def _version(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _md(text: str) -> str:
    return markdown.markdown(text.replace("\r\n", "\n"), extensions=["fenced_code", "tables", "toc", "sane_lists"])


def _mark(text: str) -> str:
    """Number every box in the source (a literal marker char in the source is dropped)."""
    n = iter(range(1 << 30))
    return _BOX.sub(lambda m: f"\ue000{next(n)}\ue001{m.group()}\ue002", _PUA.sub("", text))


class _Kinds(HTMLParser):
    """Box number -> 'row' (in a table body cell) or 'item' (first thing in a list item).
    Boxes in code, attributes or the middle of a sentence get no entry."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.kinds: dict[int, str] = {}
        self.stack: list[str] = []
        self.li_text: list[bool] = []  # per open <li>: has text come before?
        self.cells: list[str] | None = None  # cell texts of the open <tr>
        self.boxed = False  # the open <tr> holds a row box
        self.rows: list[tuple[str, ...]] = []  # cell texts of every row that holds a box

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID:
            self.stack.append(tag)
            if tag == "li":
                self.li_text.append(False)
            elif tag == "tr":
                self.cells, self.boxed = [], False
            elif tag in ("td", "th") and self.cells is not None:
                self.cells.append("")

    def handle_endtag(self, tag):
        while tag in self.stack:
            t = self.stack.pop()
            if t == "li":
                self.li_text.pop()
            elif t == "tr" and self.cells is not None:
                if self.boxed:
                    self.rows.append(tuple(c.strip() for c in self.cells))
                self.cells = None
            if t == tag:
                break

    def handle_data(self, data):
        code = "code" in self.stack or "pre" in self.stack
        if self.cells:
            self.cells[-1] += _MARK.sub(lambda m: m.group(2), data)
        for k, piece in enumerate(_MARK.split(data)):
            if k % 3 == 1 and not code:  # split yields text, number, box, text, ...
                if "td" in self.stack and "tbody" in self.stack:
                    self.kinds[int(piece)] = "row"
                    self.boxed = True
                elif self.li_text and not self.li_text[-1] and self.stack[-1] in ("li", "p"):
                    self.kinds[int(piece)] = "item"
            if k % 3 != 1 and piece.strip() and self.li_text:
                self.li_text[-1] = True


def _parse(text: str) -> _Kinds:
    p = _Kinds()
    p.feed(_md(_mark(text)))
    return p


def _kinds(text: str) -> dict[int, str]:
    return _parse(text).kinds


_EOL = re.compile(r"\r\n|\r|\n")  # markdown ends a line at any of these


def _line_of(text: str, pos: int) -> int:
    return len(_EOL.findall(text, 0, pos))


def _movable(text: str, kinds: dict[int, str]) -> set[int]:
    """Row boxes whose source line is a markdown pipe row (a raw <table> cell is clickable,
    but its row spans several lines, so it does not move)."""
    lines = _EOL.split(text)
    return {i for i, m in enumerate(_BOX.finditer(text)) if kinds.get(i) == "row"
            and "|" in (ln := lines[_line_of(text, m.start())]) and not ln.lstrip().startswith("<")}


def edit_checklist(text: str, op: str, i: int, to: int | None = None, after: bool = False) -> str:
    """'check' flips box i (☑ -> ☐, else -> ☑). 'move' puts the table row holding box i
    before/after the row holding box `to` (any table of the page). Line endings are kept.
    Raises ValueError."""
    boxes, kinds = list(_BOX.finditer(text)), _kinds(text)
    if i not in kinds:
        raise ValueError("no such box")
    if op == "check":
        m = boxes[i]
        return text[:m.start()] + ("☐" if m.group() == "☑" else "☑") + text[m.end():]
    if op != "move":
        raise ValueError("unknown op")
    movable = _movable(text, kinds)
    if i not in movable or to not in movable:
        raise ValueError("only markdown table rows move")
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", text)
    trimmed = bool(lines) and not lines[-1].endswith(("\n", "\r"))
    if trimmed:  # give the last line its neighbour's ending while it moves around
        lines[-1] += _EOL.search(lines[-2]).group() if len(lines) > 1 else "\n"
    src, dst = _line_of(text, boxes[i].start()), _line_of(text, boxes[to].start())
    if src != dst:
        row = lines.pop(src)
        dst -= src < dst
        lines.insert(dst + 1 if after else dst, row)
    out = "".join(lines)
    if trimmed:  # the file had no final line ending: drop whichever one is last now
        out = re.sub(r"(\r\n|\r|\n)\Z", "", out)
    # whatever the source looked like, every boxed row must render with exactly the same
    # cells after the move (another column count, a raw html cell, a \\| ... would change them)
    if sorted(_parse(out).rows) != sorted(_parse(text).rows):
        raise ValueError("that row would change in that table (different columns?)")
    return out


def _grant(page: str, version: str) -> str:
    """A capability for one render of one page at one version."""
    cap = secrets.token_urlsafe(16)
    with _edit_lock:
        _caps[cap] = (page, version)
        while len(_caps) > 500:
            _caps.pop(next(iter(_caps)))
    return cap


def _read_raw(idx: Path) -> str:
    return idx.read_bytes().decode("utf-8", "replace")  # no newline translation


_CHECK_JS = """(()=>{
const cap=document.querySelector('article').dataset.cap,m=location.hash.match(/^#qy=(\\d+)$/);
if(m)addEventListener('load',()=>scrollTo(0,+m[1]));
const send=d=>parent.postMessage(Object.assign({cap,y:Math.round(scrollY)},d),'*');
document.addEventListener('click',e=>{const b=e.target.closest('button.cb');if(b)send({qillqa:'check',i:+b.dataset.i})});
let drag=null,pid=null,over=null,after=false,x=0,y=0,raf=0;
const box=tr=>tr&&tr.querySelector('.grip')&&tr.querySelector('button.cb');
function mark(){if(over)over.classList.remove('drop-before','drop-after');
  const t=document.elementFromPoint(x,y);over=t&&t.closest('tr');
  if(!box(over)||over===drag){over=null;return}
  const r=over.getBoundingClientRect();after=y>r.top+r.height/2;over.classList.add(after?'drop-after':'drop-before')}
function tick(){const z=80;if(y<z)scrollBy(0,-(z-y)/3);else if(y>innerHeight-z)scrollBy(0,(y-innerHeight+z)/3);
  mark();raf=requestAnimationFrame(tick)}
function end(ok,e){if(!drag||e.pointerId!==pid)return;cancelAnimationFrame(raf);drag.classList.remove('dragging');
  if(over)over.classList.remove('drop-before','drop-after');
  if(ok&&over)send({qillqa:'move',i:+box(drag).dataset.i,to:+box(over).dataset.i,after});drag=over=pid=null}
document.addEventListener('pointerdown',e=>{const g=e.target.closest('.grip');if(!g||drag)return;e.preventDefault();
  pid=e.pointerId;drag=g.closest('tr');x=e.clientX;y=e.clientY;drag.classList.add('dragging');
  try{g.setPointerCapture(e.pointerId)}catch(_){}tick()});
document.addEventListener('pointermove',e=>{if(drag&&e.pointerId===pid){x=e.clientX;y=e.clientY}});
document.addEventListener('pointerup',e=>end(true,e));document.addEventListener('pointercancel',e=>end(false,e));
})();"""

_CHECK_CSS = """<style>
button.cb{all:unset;cursor:pointer;font-size:1.3em;line-height:1;padding:2px 4px;border-radius:6px}
button.cb:hover,button.cb:focus-visible{background:color-mix(in srgb,var(--c3) 15%,#fff);outline:2px solid var(--c3)}
.grip{cursor:grab;touch-action:none;user-select:none;color:var(--mute);padding:2px 6px 2px 0;font-size:1.2em}
td:has(>.grip){white-space:nowrap}
tr.dragging td{opacity:.45}tr.drop-before td{box-shadow:inset 0 3px 0 var(--c2)}tr.drop-after td{box-shadow:inset 0 -3px 0 var(--c2)}
</style>"""


def _render_md(title: str, text: str, cap: str = "", nonce: str = "") -> str:
    """With a capability and a nonce, boxes markdown put in table rows / list items are live."""
    kinds = _kinds(text) if cap else {}
    if kinds:
        rows: dict[int, int] = {}  # source line -> its first box: that one gets the grip
        movable = _movable(text, kinds)
        for i, m in enumerate(_BOX.finditer(text)):
            if i in movable:
                rows.setdefault(_line_of(text, m.start()), i)
        grips = set(rows.values())

        def live(m):
            i = int(m.group(1))
            if i not in kinds:
                return m.group(2)
            g = '<span class="grip" title="Drag to move">⠿</span>' if i in grips else ""
            return f'{g}<button class="cb" data-i="{i}" title="Click to mark">{m.group(2)}</button>'
        parts = re.split(r"(<[^>]*>)", _md(_mark(text)))  # markers inside a tag are just the glyph
        body = "".join(_MARK.sub(lambda m: m.group(2), p) if p.startswith("<") else _MARK.sub(live, p)
                       for p in parts)
        body = _CHECK_CSS + body + f'<script nonce="{nonce}">{_CHECK_JS}</script>'
    else:
        body = _md(text)
    # ![x](clip.mp4) -> video / audio player
    body = re.sub(rf'<img alt="([^"]*)" src="([^"]+\.(?:{_VIDEO}))"\s*/?>',
                  r'<video controls preload="metadata" src="\2" title="\1"></video>', body, flags=re.I)
    body = re.sub(rf'<img alt="([^"]*)" src="([^"]+\.(?:{_AUDIO}))"\s*/?>',
                  r'<audio controls preload="metadata" src="\2" title="\1"></audio>', body, flags=re.I)
    # links leave the sandboxed page in a new tab; in-page #anchors stay put
    body = re.sub(r'<a href="(?!#)', '<a target="_blank" href="', body)
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    attr = f' data-cap="{cap}"' if kinds else ""
    return _shell(title, f'<article{attr}>{body}</article>')


@app.get("/view")
def view(path: str = ""):
    """Open the page (session required) at its signed URL."""
    try:
        d = _safe(path)
    except ValueError as e:
        _bad(e)
    if d == PAGES or not _index(d):
        _bad(ValueError(f"no page at {path!r}"))
    return RedirectResponse(signed_url(_rel(d)), status_code=303)


@app.get("/f/{sig}/{rest:path}")
def signed_file(sig: str, rest: str, req: Request):
    parts = [p for p in rest.split("/") if p]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        _bad(ValueError("invalid path"))
    # signed folder = the longest prefix whose signature matches (never the root)
    scope = next((PAGES.joinpath(*parts[:n]) for n in range(len(parts), 0, -1)
                  if hmac.compare_digest(sig, _sig("/".join(parts[:n])))), None)
    try:
        target = _safe(rest)
    except ValueError as e:
        _bad(e)
    if scope is None or (target != scope.resolve() and scope.resolve() not in target.parents):
        _bad(ValueError("invalid signature"))
    if target.is_dir():
        if not req.url.path.endswith("/"):  # relative links need the trailing slash
            return RedirectResponse(req.url.path + "/", status_code=307)
        idx = _index(target)
        if not idx:
            _bad(ValueError("no page here"))
        text = _read_raw(idx)
        if idx.suffix == ".html":
            body, csp = text, _HTML_SANDBOX
        elif _kinds(text):  # checklist: only our nonce'd script runs, never raw <script> in the md
            nonce, cap = secrets.token_urlsafe(16), _grant(_rel(target), _version(text))
            body, csp = _render_md(_title(target), text, cap, nonce), f"{_MD_LIVE_SANDBOX}; script-src 'nonce-{nonce}'"
        else:
            body, csp = _render_md(_title(target), text), _MD_SANDBOX
        return Response(body, media_type="text/html; charset=utf-8",
                        headers={**_SAFE_HEADERS, "Cache-Control": "no-store", "Content-Security-Policy": csp})
    if not target.is_file():
        _bad(ValueError("not found"))
    r = FileResponse(target, headers={**_SAFE_HEADERS, "Cache-Control": "private"})
    if r.media_type != "application/pdf":  # Chrome refuses to render PDFs in a sandbox
        r.headers["Content-Security-Policy"] = (_HTML_SANDBOX if r.media_type.startswith("text/html")
                                                 else "sandbox")
    return r


CSS = """
:root{--bg:#f4f6fb;--fg:#1d2433;--mute:#5d6680;--line:#e2e6f0;--card:#fff;--acc:#5b3fd0;
--c1:#0f7c8c;--c2:#c2502e;--c3:#6d4bd1;--c4:#c98300;--c5:#2f8f5b;color-scheme:light}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,sans-serif}
a{color:var(--acc);text-underline-offset:3px}code,pre{font-family:ui-monospace,monospace;font-size:.9em}
article{max-width:880px;margin:24px auto;padding:36px 44px;background:var(--card);border-radius:18px;
box-shadow:0 1px 3px #1d243314,0 12px 32px #1d24330d}
article h1{font-size:2.3em;line-height:1.15;margin:0 0 .6em;padding-bottom:.35em;
background:linear-gradient(90deg,var(--c2),var(--c3) 55%,var(--c1));-webkit-background-clip:text;background-clip:text;color:transparent;
border-bottom:4px solid;border-image:linear-gradient(90deg,var(--c2),var(--c3),var(--c1)) 1}
article h2{--c:var(--c1);margin:1.8em 0 .7em;padding:8px 14px;border-left:6px solid var(--c);border-radius:8px;
background:color-mix(in srgb,var(--c) 11%,#fff);color:color-mix(in srgb,var(--c) 75%,#000)}
article h2:nth-of-type(5n+2){--c:var(--c2)}article h2:nth-of-type(5n+3){--c:var(--c3)}
article h2:nth-of-type(5n+4){--c:var(--c4)}article h2:nth-of-type(5n+5){--c:var(--c5)}
article h3{color:var(--c3);margin:1.6em 0 .5em;padding-bottom:4px;border-bottom:2px dotted color-mix(in srgb,var(--c3) 35%,#fff)}
article li::marker{color:var(--c2)}
code{background:#fff0ea;color:#a8401c;padding:1px 6px;border-radius:5px}
pre{background:#f5f7ff;border:1px solid #e3e7fb;border-left:4px solid var(--c3);padding:14px 16px;overflow:auto;border-radius:10px}
pre code{background:none;color:#1d2433;padding:0}
blockquote{margin:1em 0;padding:10px 16px;background:#ecf8f6;border-left:4px solid var(--c1);border-radius:8px;color:#174a52}
blockquote p{margin:.4em 0}
table{border-collapse:separate;border-spacing:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:1em 0}
th{background:linear-gradient(90deg,var(--c1),var(--c3));color:#fff;text-align:left;font-weight:600}
.tw{overflow-x:auto;margin:1em 0}.tw table{margin:0}
td,th{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:#f7f8fe}
hr{border:0;height:3px;background:linear-gradient(90deg,var(--c2),var(--c3),var(--c1));border-radius:3px}
img,video{max-width:100%;border-radius:10px;box-shadow:0 4px 16px #1d24331f}audio{width:100%}
@media (max-width:640px){article{margin:0;padding:22px 16px;border-radius:0;box-shadow:none}}
.login{max-width:340px;margin:18vh auto;display:grid;gap:12px;padding:28px;background:var(--card);border-radius:16px;
box-shadow:0 12px 32px #1d24331a}
.login h1{margin:0;background:linear-gradient(90deg,var(--c2),var(--c3),var(--c1));-webkit-background-clip:text;background-clip:text;color:transparent}
input,button{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
button{cursor:pointer;background:linear-gradient(90deg,var(--c3),var(--c1));color:#fff;border:0}.err{color:var(--c2);margin:0}
"""


def _shell(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>")


INDEX = """
<header><a href="#" class="brand">qillqa</a>
<input id="q" type="search" placeholder="Search by meaning…" autocomplete="off">
<label id="scope" hidden title="Search only this folder and its subpages"><input type="checkbox" id="only"> only in <b id="scopename"></b></label>
<a href="/logout" class="out">sign out</a></header>
<main><aside><button id="toggle">Pages</button><nav id="tree"></nav></aside>
<section><div id="results" hidden></div><div id="crumb"></div>
<iframe id="page" sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals" title="page"></iframe>
<p id="empty">Pick a page or search for something.</p></section></main>
<style>
header{display:flex;gap:12px;align-items:center;padding:10px 16px;background:var(--card);
border-bottom:3px solid;border-image:linear-gradient(90deg,var(--c2),var(--c3),var(--c1)) 1}
.brand{font-weight:800;font-size:1.15em;text-decoration:none;background:linear-gradient(90deg,var(--c2),var(--c3));
-webkit-background-clip:text;background-clip:text;color:transparent}
#q{flex:1;min-width:0;background:var(--bg)}#q:focus{outline:2px solid var(--c3);background:#fff}.out{color:var(--mute);font-size:.9em}
#scope{font-size:.85em;color:var(--mute);white-space:nowrap;max-width:30vw;overflow:hidden;text-overflow:ellipsis;cursor:pointer}
#scope b{color:var(--c1)}#scope input{accent-color:var(--c3);vertical-align:middle}
main{display:grid;grid-template-columns:280px 1fr;height:calc(100vh - 62px)}
aside{background:#fbfcff;border-right:1px solid var(--line);overflow:auto;padding:10px}#toggle{display:none}
nav ul{list-style:none;margin:0;padding-left:14px}nav>ul{padding-left:0}
nav a{display:block;padding:4px 8px;border-radius:6px;color:var(--fg);text-decoration:none;overflow-wrap:anywhere}
nav a:hover{background:color-mix(in srgb,var(--c3) 8%,#fff)}
nav a.sec{color:var(--c1);font-weight:600}
nav a.on{background:color-mix(in srgb,var(--c3) 14%,#fff);color:color-mix(in srgb,var(--c3) 80%,#000);font-weight:600;
box-shadow:inset 3px 0 0 var(--c3)}
section{display:flex;flex-direction:column;min-width:0}iframe{flex:1;border:0;width:100%;background:var(--bg)}
#crumb{padding:6px 16px;color:var(--c1);font-size:.85em;font-weight:600;overflow-wrap:anywhere}#empty{padding:24px 16px;color:var(--mute)}
#results{padding:8px 16px;background:var(--card);border-bottom:1px solid var(--line);max-height:45vh;overflow:auto}
.hit{display:block;padding:10px 12px;margin:6px 0;border-radius:10px;border:1px solid var(--line);color:var(--fg);text-decoration:none}
.hit:hover{border-color:var(--c3);background:color-mix(in srgb,var(--c3) 5%,#fff)}
.hit b{color:var(--c3)}.hit small{color:var(--c1);display:block}
@media (max-width:760px){header{flex-wrap:wrap}#scope{order:3;max-width:100%}
main{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid var(--line)}
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
let scope='';
async function show(){
  const p=decodeURIComponent(location.hash.slice(1));mark();
  if(!scope){$('#scopename').textContent=p;$('#scope').hidden=!p}
  if(!p){$('#page').hidden=true;$('#empty').hidden=false;$('#crumb').textContent='';return}
  $('#crumb').textContent=p.split('/').join(' / ');$('#empty').hidden=true;$('#page').hidden=false;
  $('#page').src='/view?path='+encodeURIComponent(p);$('nav').classList.remove('open');
}
let t;
$('#q').addEventListener('input',()=>{clearTimeout(t);t=setTimeout(find,400)});
async function find(){
  const q=$('#q').value.trim(),box=$('#results');
  if(!q){box.hidden=true;return}
  box.hidden=false;box.replaceChildren(mk('small',{textContent:'Searching… (the first search loads the model)'}));
  const hits=await api('/api/search?q='+encodeURIComponent(q)+(scope?'&path='+encodeURIComponent(scope):''));
  box.replaceChildren(...(hits.length?hits.map(h=>mk('a',{className:'hit',href:'#'+h.path},
    mk('b',{textContent:h.title}),mk('small',{textContent:h.path+' · '+h.score}),
    mk('span',{textContent:h.snippet}))):[mk('small',{textContent:scope?'No results in '+scope:'No results'})]));
}
$('#only').onchange=()=>{scope=$('#only').checked?decodeURIComponent(location.hash.slice(1)):'';show();find()};
$('#toggle').onclick=()=>$('nav').classList.toggle('open');
// checklist pages post {qillqa: check|move} here; only the shown page, only its own iframe
addEventListener('message',async e=>{
  const f=$('#page'),d=e.data,p=decodeURIComponent(location.hash.slice(1));
  if(e.source!==f.contentWindow||!d||!['check','move'].includes(d.qillqa)||!p)return;
  const r=await fetch('/api/page/edit',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:p,op:d.qillqa,i:d.i,to:d.to,after:!!d.after,cap:d.cap})});
  if(r.status===401){location='/login';return}
  if(decodeURIComponent(location.hash.slice(1))!==p)return;  // navigated away meanwhile
  if(!r.ok&&r.status!==409)$('#crumb').textContent='⚠ '+((await r.json()).detail||r.status);
  f.src='/view?path='+encodeURIComponent(p)+'&r='+Date.now()+'#qy='+(d.y|0);
});
addEventListener('hashchange',show);loadTree().then(show);
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _shell("qillqa", INDEX)
