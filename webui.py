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
from markdown.extensions.toc import slugify
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

import embedder
import server
from server import MODEL, PAGES, TOKEN, VAR, _children, _index, _rel, _safe, _title, read_page, tree

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
    if req.url.path in ("/health", "/login") or req.url.path.startswith(("/f/", "/s/")) or _authed(req):
        return await call_next(req)
    if req.url.path.startswith(("/api/", "/view")):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/health")
def health():
    return {"app": "qillqa", "ok": True}


@app.get("/login", response_class=HTMLResponse)
def login_page(req: Request, error: str = ""):
    msg = '<p class="err">Wrong token</p>' if error else ""
    return _shell("Sign in", f"""<form method="post" action="/login" class="login">
<h1>qillqa</h1>{msg}<input type="password" name="token" placeholder="Token" autofocus required>
<button>Sign in</button></form>""", _theme(req))


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
                                 None if d.get("to") is None else int(d["to"]), bool(d.get("after")),
                                 None if d.get("line") is None else int(d["line"]))
            fd, tmp = tempfile.mkstemp(dir=idx.parent, prefix=".edit-", suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(new.encode())
            os.replace(tmp, idx)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    threading.Thread(target=sync, daemon=True).start()
    return {"ok": True}


def _page_index(path: str) -> tuple[Path, Path]:
    d = _safe(path)
    idx = _index(d) if d != PAGES else None
    if not idx:
        raise ValueError(f"no page at {path!r}")
    if idx.is_symlink():  # its target may live outside the notebook
        raise ValueError(f"{path!r}: the page file is a link, edit it where it lives")
    return d, idx


def _folder_files(d: Path) -> list[dict]:
    """Every file the page folder holds, down its non-page subfolders (assets/…); subpages are theirs."""
    base, out = signed_url(_rel(d)), []
    for root, dirs, files in os.walk(d):
        r = Path(root)
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and not _index(r / x))
        for f in sorted(files):
            p = r / f
            if f.startswith(".") or (r == d and f in server.INDEX_NAMES) or p.is_symlink():
                continue
            rel, st = p.relative_to(d).as_posix(), p.stat()
            out.append({"name": rel, "size": st.st_size, "mtime": st.st_mtime,
                        "url": base + "/".join(urllib.parse.quote(s) for s in rel.split("/"))})
    return out


MAX_SOURCE = 5_000_000  # bytes of the index file


def _source_text(idx: Path) -> tuple[str, str]:
    """(text, readonly reason). The version is taken on this text. A file the editor can't write
    back byte for byte (invalid UTF-8, mixed line endings) is shown but never saved."""
    raw = idx.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", "replace"), "not valid UTF-8"
    crlf = text.count("\r\n")
    if crlf != text.count("\r"):  # a lone CR: the textarea turns it into LF
        return text, "lone carriage returns"
    return text, "mixed line endings" if 0 < crlf < text.count("\n") else ""


@app.get("/api/source")
def api_source(path: str = ""):
    """The page's index.md/html (line endings as \\n, like a textarea) plus the files in its folder."""
    try:
        d, idx = _page_index(path)
        text, readonly = _source_text(idx)
        files = _folder_files(d)
    except (ValueError, OSError) as e:  # OSError: moved or deleted while we read
        _bad(e)
    return {"path": _rel(d), "format": idx.suffix[1:], "content": text.replace("\r\n", "\n"),
            "version": _version(text), "readonly": readonly, "files": files}


@app.post("/api/source")
async def api_source_save(req: Request):
    """{path, content, version}: 409 if the page changed since that version (MCP or another tab).
    The file keeps its CRLF line endings if it had them."""
    if not req.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "json only")
    # ponytail: Content-Length only; a chunked body is buffered before the byte check below
    if int(req.headers.get("content-length") or 0) > 2 * MAX_SOURCE:
        raise HTTPException(413, "content must be at most 5 MB")
    d = await req.json()
    content = d.get("content")
    if not isinstance(content, str):
        raise HTTPException(400, "content must be text")
    content = content.replace("\r\n", "\n")
    try:
        with server.page_lock():  # resolve inside the lock: the MCP may move or delete the page
            page, idx = _page_index(str(d.get("path", "")))
            old, readonly = _source_text(idx)
            if readonly:
                raise HTTPException(422, f"the page has {readonly}; edit it where it lives")
            if _version(old) != d.get("version"):
                return JSONResponse({"error": "the page changed since you opened it"}, status_code=409)
            if "\r\n" in old:
                content = content.replace("\n", "\r\n")
            data = content.encode()
            if len(data) > MAX_SOURCE:
                raise HTTPException(413, "content must be at most 5 MB")
            fd, tmp = tempfile.mkstemp(dir=idx.parent, prefix=".edit-", suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, idx)
    except (ValueError, OSError) as e:
        _bad(e)
    threading.Thread(target=sync, daemon=True).start()
    return {"ok": True, "version": _version(content)}


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


_HTML_SANDBOX = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals allow-top-navigation-by-user-activation"
_MD_SANDBOX = "sandbox allow-popups allow-popups-to-escape-sandbox allow-top-navigation-by-user-activation"
_MD_LIVE_SANDBOX = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox allow-top-navigation-by-user-activation"
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
_PUA = re.compile("[\ue000-\ue004]")
_HDR = re.compile("\ue003(\\d+)\ue004")  # header row of a table: its separator line
_MARK = re.compile(f"\ue000(\\d+)\ue001([{BOXES}])\ue002")
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_caps: dict[str, tuple[str, str]] = {}  # capability -> (page, version); newest last
_edit_lock = threading.Lock()


def _version(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _slugify(value: str, separator: str) -> str:
    """Heading ids as if there were no box numbers or header tags in the text."""
    return slugify(_HDR.sub("", _MARK.sub(lambda m: m.group(2), value)), separator)


def _md(text: str) -> str:
    return markdown.markdown(text.replace("\r\n", "\n"), extensions=["fenced_code", "tables", "toc", "sane_lists", "attr_list", "md_in_html"],
                             extension_configs={"toc": {"slugify": _slugify}})


def _mark(text: str, headers: set[int] | None = None) -> str:
    """Number every box in the source (a literal marker char in the source is dropped);
    with `headers`, also tag those table header lines (see _headers)."""
    n, text = iter(range(1 << 30)), _PUA.sub("", text)
    return _BOX.sub(lambda m: f"\ue000{next(n)}\ue001{m.group()}\ue002",
                    _headers(text, headers) if headers else text)


class _Kinds(HTMLParser):
    """Box number -> 'row' (in a table body cell) or 'item' (first thing in a list item).
    Boxes in code, attributes or the middle of a sentence get no entry."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.kinds: dict[int, str] = {}
        self.stack: list[str] = []
        self.li_text: list[bool] = []  # per open <li>: has text come before?
        self.md_table: list[bool] = []  # per open <table>: built by markdown (its header is tagged)
        self.cells: list[str] | None = None  # cell texts of the open <tr>
        self.boxed = False  # the open <tr> holds a row box
        self.rows: list[tuple[str, ...]] = []  # cell texts of every row that holds a box

    def handle_starttag(self, tag, attrs):
        if self.cells and tag not in ("td", "th", "tr"):
            self.cells[-1] += self.get_starttag_text()  # an image or link is part of the cell
        if tag not in _VOID:
            self.stack.append(tag)
            if tag == "li":
                self.li_text.append(False)
            elif tag == "table":
                self.md_table.append(False)
            elif tag == "tr":
                self.cells, self.boxed = [], False
            elif tag in ("td", "th") and self.cells is not None:
                self.cells.append("")

    def handle_endtag(self, tag):
        if self.cells and tag not in ("td", "th", "tr"):
            self.cells[-1] += f"</{tag}>"
        while tag in self.stack:
            t = self.stack.pop()
            if t == "li":
                self.li_text.pop()
            elif t == "table" and self.md_table:
                self.md_table.pop()
            elif t == "tr" and self.cells is not None:
                if self.boxed:
                    self.rows.append(tuple(c.strip() for c in self.cells))
                self.cells = None
            if t == tag:
                break

    def handle_data(self, data):
        code = "code" in self.stack or "pre" in self.stack
        if "th" in self.stack and self.md_table and _HDR.search(data):
            self.md_table[-1] = True
        if self.cells:
            self.cells[-1] += _HDR.sub("", _MARK.sub(lambda m: m.group(2), data))
        for k, piece in enumerate(_MARK.split(data)):
            if k % 3 == 1 and not code:  # split yields text, number, box, text, ...
                if "td" in self.stack and "tbody" in self.stack:
                    md = bool(self.md_table) and self.md_table[-1]
                    self.kinds[int(piece)] = "row" if md else "cell"  # a raw <table> cell clicks, never moves
                    self.boxed = True
                elif self.li_text and not self.li_text[-1] and self.stack[-1] in ("li", "p"):
                    self.kinds[int(piece)] = "item"
            if k % 3 != 1 and piece.strip() and self.li_text:
                self.li_text[-1] = True


def _html(text: str) -> str:
    """Markdown with numbered boxes and tagged table headers. A header tag that would change
    how the page renders is left out (that table then gets no drop target and no grips)."""
    plain = _md(_mark(text))
    clean = lambda tags: _HDR.sub("", _md(_mark(text, tags))) == plain  # noqa: E731
    ok = {n for n in _header_lines(_PUA.sub("", text)) if clean({n})}
    tagged = _md(_mark(text, ok)) if ok else plain
    return tagged if _HDR.sub("", tagged) == plain else plain


def _parse(text: str) -> _Kinds:
    p = _Kinds()
    p.feed(_html(text))
    return p


def _kinds(text: str) -> dict[int, str]:
    return _parse(text).kinds


_EOL = re.compile(r"\r\n|\r|\n")  # markdown ends a line at any of these


def _line_of(text: str, pos: int) -> int:
    return len(_EOL.findall(text, 0, pos))


def _movable(text: str, kinds: dict[int, str]) -> set[int]:
    """Boxes in rows of tables markdown built (see _Kinds); those rows move."""
    return {i for i, k in kinds.items() if k == "row"}


_SEPLINE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


def _header_lines(text: str) -> list[int]:
    """Separator lines of pipe-table candidates (markdown has the final word, see _html)."""
    lines = _EOL.split(text)
    return [n for n in range(1, len(lines)) if _SEPLINE.match(lines[n]) and "|" in lines[n - 1]]


def _headers(text: str, only: set[int] | None = None) -> str:
    """Tag the header line of each pipe table (or of those in `only`) with its separator's
    line number (\ue003n\ue004), inside the first cell: after a list marker, pipe and spaces."""
    lines, seps = _EOL.split(text), _EOL.findall(text) + [""]
    for n in _header_lines(text):
        if only is None or n in only:
            h = lines[n - 1]
            k = re.match(r"\s*(?:(?:[-*+]|\d+[.)])\s+)?\|?[ \t]*", h).end()
            lines[n - 1] = f"{h[:k]}\ue003{n}\ue004{h[k:]}"
    return "".join(a + b for a, b in zip(lines, seps))


def edit_checklist(text: str, op: str, i: int, to: int | None = None, after: bool = False,
                   line: int | None = None) -> str:
    """'check' flips box i (☑ -> ☐, else -> ☑). 'move' puts the table row holding box i
    before/after the row holding box `to`, or, with `line` (a table's separator line), first
    in that table, even an empty one. Line endings are kept. Raises ValueError."""
    boxes, kinds = list(_BOX.finditer(text)), _kinds(text)
    if i not in kinds:
        raise ValueError("no such box")
    if op == "check":
        m = boxes[i]
        return text[:m.start()] + ("☐" if m.group() == "☑" else "☑") + text[m.end():]
    if op != "move":
        raise ValueError("unknown op")
    movable = _movable(text, kinds)
    if i not in movable or (line is None and to not in movable):
        raise ValueError("only markdown table rows move")
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", text)
    trimmed = bool(lines) and not lines[-1].endswith(("\n", "\r"))
    if trimmed:  # give the last line its neighbour's ending while it moves around
        lines[-1] += _EOL.search(lines[-2]).group() if len(lines) > 1 else "\n"
    src = _line_of(text, boxes[i].start())
    if line is not None:  # first row of the table whose separator is `line`
        if not (0 < line < len(lines) and _SEPLINE.match(lines[line].rstrip("\r\n"))):
            raise ValueError("no table there")
        dst, after = line, True
    else:
        dst = _line_of(text, boxes[to].start())
    if src != dst:
        row = lines.pop(src)
        dst -= src < dst
        lines.insert(dst + 1 if after else dst, row)
    out = "".join(lines)
    if trimmed:  # the file had no final line ending: drop whichever one is last now
        out = re.sub(r"(\r\n|\r|\n)\Z", "", out)
    # whatever the source looked like, every boxed row must render with exactly the same
    # cells after the move (another column count, a raw html cell, a \\| ... would change them),
    # and the rest of the page must render exactly as before (joined line endings, ...)
    if sorted(_parse(out).rows) != sorted(_parse(text).rows) or _rest(out) != _rest(text):
        raise ValueError("that row would change in that table (different columns?)")
    return out


def _rest(text: str) -> str:
    """The rendered page without the rows that hold a box (box numbers dropped)."""
    html_ = re.sub(r"<tr>(?:(?!</tr>).)*?\ue000.*?</tr>\s*", "", _md(_mark(text)), flags=re.S)
    html_ = re.sub(r"<tr>(?:\s*<td[^>]*>\s*</td>)+\s*</tr>\s*", "", html_)  # an emptied table's filler row
    return _MARK.sub(lambda m: m.group(2), html_)


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
const box=tr=>tr&&tr.querySelector('.grip')&&tr.querySelector('button.cb'),hdr=tr=>tr&&tr.querySelector('.hdr');
function mark(){if(over)over.classList.remove('drop-before','drop-after');
  const t=document.elementFromPoint(x,y);over=t&&t.closest('tr');
  if(over===drag||!(box(over)||hdr(over))){over=null;return}
  const r=over.getBoundingClientRect();after=!!hdr(over)||y>r.top+r.height/2;over.classList.add(after?'drop-after':'drop-before')}
function tick(){const z=80;if(y<z)scrollBy(0,-(z-y)/3);else if(y>innerHeight-z)scrollBy(0,(y-innerHeight+z)/3);
  mark();raf=requestAnimationFrame(tick)}
function end(ok,e){if(!drag||e.pointerId!==pid)return;cancelAnimationFrame(raf);drag.classList.remove('dragging');
  if(over)over.classList.remove('drop-before','drop-after');
  if(ok&&over)send(hdr(over)?{qillqa:'move',i:+box(drag).dataset.i,line:+hdr(over).dataset.l}
    :{qillqa:'move',i:+box(drag).dataset.i,to:+box(over).dataset.i,after});drag=over=pid=null}
document.addEventListener('pointerdown',e=>{const g=e.target.closest('.grip');if(!g||drag)return;e.preventDefault();
  pid=e.pointerId;drag=g.closest('tr');x=e.clientX;y=e.clientY;drag.classList.add('dragging');
  try{g.setPointerCapture(e.pointerId)}catch(_){}tick()});
document.addEventListener('pointermove',e=>{if(drag&&e.pointerId===pid){x=e.clientX;y=e.clientY}});
document.addEventListener('pointerup',e=>end(true,e));document.addEventListener('pointercancel',e=>end(false,e));
})();"""

_CHECK_CSS = """<style>
button.cb{all:unset;cursor:pointer;font-size:1.3em;line-height:1;padding:2px 4px;border-radius:6px}
button.cb:hover,button.cb:focus-visible{background:color-mix(in srgb,var(--c3) 15%,var(--card));outline:2px solid var(--c3)}
.grip{cursor:grab;touch-action:none;user-select:none;color:var(--mute);padding:2px 6px 2px 0;font-size:1.2em}
td:has(>.grip){white-space:nowrap}
tr.dragging td{opacity:.45}tr.drop-before :is(td,th){box-shadow:inset 0 3px 0 var(--c2)}tr.drop-after :is(td,th){box-shadow:inset 0 -3px 0 var(--c2)}
</style>"""


def _render_md(title: str, text: str, cap: str = "", nonce: str = "", theme: str = "light", link=None) -> str:
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
        parts = re.split(r"(<[^>]*>)", _html(text))  # markers inside a tag are just the glyph
        body, in_th = [], False
        for p in parts:
            if p.startswith("<"):
                in_th = p.startswith("<th") or (in_th and not p.startswith("</th"))
                body.append(_HDR.sub("", _MARK.sub(lambda m: m.group(2), p)))
            else:
                hdr = (lambda m: f'<span class="hdr" data-l="{m.group(1)}"></span>') if in_th else ""
                body.append(_HDR.sub(hdr, _MARK.sub(live, p)))
        body = "".join(body)
        body = _CHECK_CSS + body + f'<script nonce="{nonce}">{_CHECK_JS}</script>'
    else:
        body = _md(text)
    # ![x](clip.mp4) -> video / audio player
    body = re.sub(rf'<img alt="([^"]*)" src="([^"]+\.(?:{_VIDEO}))"\s*/?>',
                  r'<video controls preload="metadata" src="\2" title="\1"></video>', body, flags=re.I)
    body = re.sub(rf'<img alt="([^"]*)" src="([^"]+\.(?:{_AUDIO}))"\s*/?>',
                  r'<audio controls preload="metadata" src="\2" title="\1"></audio>', body, flags=re.I)
    # [x](#other/page) or [x](/#other/page) opens that notebook page in place (the shell
    # follows the hash); a #heading of this page stays an in-page anchor
    body = re.sub(r'<a href="/?#([^"]+)"', lambda m: _page_link(m, body, link), body)
    # other links leave the sandboxed page in a new tab
    body = re.sub(r'<a href="(?!#)', '<a target="_blank" href="', body)
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    attr = f' data-cap="{cap}"' if kinds else ""
    return _shell(title, f'<article{attr}>{body}</article>', theme)


def _page_link(m: re.Match, body: str, link=None) -> str:
    """link: in a share, maps a page path to its URL inside the share, or None (not shared)."""
    frag = m.group(1)
    if f'id="{frag}"' in body:
        return m.group(0)
    try:
        d = _safe(urllib.parse.unquote(html.unescape(frag)))
    except ValueError:
        return m.group(0)
    if d == PAGES or not _index(d):
        return m.group(0)
    if link:
        href = link(_rel(d))
        return f'<a href="{html.escape(href)}"' if href else '<a class="off" title="Not shared"'
    return f'<a target="_top" href="/#{frag}"'


@app.get("/api/export")
def api_export(path: str = "", subpages: bool = False):
    """One self-contained .html (exporter.py). Always a download: its scripts must never run at
    the qillqa origin, where they would have the session cookie."""
    import exporter
    try:
        name, doc, report = exporter.build(path, subpages)
    except ValueError as e:
        _bad(e)
    return Response(doc, media_type="text/html; charset=utf-8", headers={
        **_SAFE_HEADERS, "Cache-Control": "no-store", "Content-Security-Policy": "sandbox",
        "Content-Disposition": f"attachment; filename=\"{name.encode('ascii', 'replace').decode().replace(chr(34), '_')}\"; "
                               f"filename*=UTF-8''{urllib.parse.quote(name)}",
        "X-Qillqa-Offline": "yes" if report["offline"] else "no"})


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
    return _serve(target, req)


def _serve(target: Path, req: Request, live: bool = True, link=None):
    """A page folder (rendered, sandboxed) or one of its files. live=False: read-only
    (no checklist capability), for share links."""
    if target.is_dir():
        if not req.url.path.endswith("/"):  # relative links need the trailing slash
            return RedirectResponse(req.url.path + "/", status_code=307)
        idx = _index(target)
        if not idx:
            _bad(ValueError("no page here"))
        text = _read_raw(idx)
        if idx.suffix == ".html":
            body, csp = text, _HTML_SANDBOX
        elif live and _kinds(text):  # checklist: only our nonce'd script runs, never raw <script> in the md
            nonce, cap = secrets.token_urlsafe(16), _grant(_rel(target), _version(text))
            body, csp = (_render_md(_title(target), text, cap, nonce, _theme(req)),
                         f"{_MD_LIVE_SANDBOX}; script-src 'nonce-{nonce}'")
        else:
            body, csp = _render_md(_title(target), text, theme=_theme(req), link=link), _MD_SANDBOX
        return Response(body, media_type="text/html; charset=utf-8",
                        headers={**_SAFE_HEADERS, "Cache-Control": "no-store", "Content-Security-Policy": csp})
    if not target.is_file():
        _bad(ValueError("not found"))
    # The page runs at an opaque origin, so its own .glb/.json (fetch) and its images and
    # videos used as WebGL textures are cross-origin reads. The signed URL is already the
    # capability (anyone holding it can GET the bytes); CORS only lets the page's JS read them.
    r = FileResponse(target, headers={**_SAFE_HEADERS, "Cache-Control": "private" if live else "no-store",
                                      "Access-Control-Allow-Origin": "*"})
    if r.media_type != "application/pdf":  # Chrome refuses to render PDFs in a sandbox
        r.headers["Content-Security-Policy"] = (_HTML_SANDBOX if r.media_type.startswith("text/html")
                                                 else "sandbox")
    return r



# ---------------------------------------------------------------- share links
# /s/<token>            a minimal read-only viewer (no tree, search, edit, export)
# /s/<token>/p/<page>/  the page and its files, checked against the link on EVERY request
# The token is checked (expiry, revocation) for the page and for each of its files, so a
# link stops working at once; nothing here ever returns a permanent /f/ signature.

_GONE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Enlace no disponible</title>
<style>body{font:16px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:90vh;margin:0;background:#f4f6fb;color:#1d2433}
div{max-width:380px;padding:28px;background:#fff;border-radius:14px;text-align:center}</style>
<div><h1 style="font-size:1.3em">Este enlace ya no está disponible</h1><p>Caducó o fue revocado. Pide un enlace nuevo a quien te lo compartió.</p></div>"""
_SHARE_HEADERS = {**_SAFE_HEADERS, "Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"}


def _real_index(d: Path) -> Path | None:
    """The page's index only if it is a real file of that folder (a symlinked index shows
    another page's content)."""
    idx = _index(d)
    return idx if idx and not idx.is_symlink() and idx.resolve().parent == d.resolve() else None


def _share_pages(share: dict) -> list[tuple[str, str, int]]:
    """(path, title, depth) the share opens, walking real folders only (no symlinks)."""
    scope = _safe(share["path"])
    out = [(share["path"], _title(scope), 0)]

    def walk(d: Path, depth: int):
        for c in _children(d):
            if c.is_symlink() or not _in_share(share, c.resolve()):
                continue
            real = _real_index(c)
            if real:
                out.append((_rel(c), _title(c), depth))
            walk(c, depth + (1 if real else 0))
    if share["subpages"]:
        walk(scope, 1)
    return out


def _share_link(share: dict, token: str):
    """Notebook links inside a shared page: to a shared page -> its URL in the share."""
    def link(rel: str) -> str | None:
        inside = rel == share["path"] or (share["subpages"] and rel.startswith(share["path"] + "/"))
        if not inside:
            return None
        try:
            d = _safe(rel)
        except ValueError:
            return None
        if not _in_share(share, d) or not _real_index(d):
            return None
        return f"/s/{token}/p/" + "".join(urllib.parse.quote(seg) + "/" for seg in rel.split("/"))
    return link


def _in_share(share: dict, target: Path) -> bool:
    """Inside the shared page; without subpages, no folder below it that is a page."""
    scope = _safe(share["path"])
    if target != scope and scope not in target.parents:
        return False
    if share["subpages"]:
        return True
    d = target if target.is_dir() else target.parent
    while d != scope:
        if _index(d):
            return False
        d = d.parent
    return True


@app.get("/s/{token}")
def share_view(token: str, req: Request):
    share = server.share_resolve(token, count=True)
    if not share:
        return HTMLResponse(_GONE, status_code=410, headers=_SHARE_HEADERS)
    if not _real_index(_safe(share["path"])):
        return HTMLResponse(_GONE, status_code=410, headers=_SHARE_HEADERS)
    pages = _share_pages(share)
    base = f"/s/{urllib.parse.quote(token)}/p/"
    src = lambda rel: base + "".join(urllib.parse.quote(seg) + "/" for seg in rel.split("/"))
    exp = time.strftime("%d/%m/%Y %H:%M", time.localtime(share["expires"]))
    opts = "".join(f'<option value="{html.escape(src(p))}">{"· " * d}{html.escape(t)}</option>' for p, t, d in pages)
    nav = (f'<select id="nav" aria-label="Página">{opts}</select>' if len(pages) > 1 else "")
    nonce = secrets.token_urlsafe(16)
    title = html.escape(pages[0][1])
    body = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>{title}</title><style>
html,body{{margin:0;height:100%;background:#f4f6fb;font:14px/1.4 system-ui,sans-serif;color:#1d2433}}
body{{display:flex;flex-direction:column}}
header{{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:8px 16px;background:#fff;border-bottom:1px solid #e2e6f0}}
header b{{font-size:15px}} header span{{color:#5d6680;font-size:12.5px}}
select{{font:inherit;padding:4px 8px;border:1px solid #e2e6f0;border-radius:7px;background:#fff;max-width:100%}}
iframe{{flex:1;width:100%;border:0;background:#fff}}
</style></head><body>
<header><b>{title}</b><span>Solo lectura · el enlace caduca el {exp}</span>{nav}</header>
<iframe id="pg" src="{html.escape(src(pages[0][0]))}" sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals" allow="fullscreen" referrerpolicy="no-referrer"></iframe>
<script nonce="{nonce}">const n=document.getElementById('nav');if(n)n.onchange=()=>document.getElementById('pg').src=n.value;</script>
</body></html>"""
    csp = (f"default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-{nonce}'; frame-src 'self'; "
           "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
    return HTMLResponse(body, headers={**_SHARE_HEADERS, "Content-Security-Policy": csp})


@app.get("/s/{token}/p/{rest:path}")
def share_file(token: str, rest: str, req: Request):
    share = server.share_resolve(token)
    if not share:
        return HTMLResponse(_GONE, status_code=410, headers=_SHARE_HEADERS)
    parts = [p for p in rest.split("/") if p]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        _bad(ValueError("invalid path"))
    try:
        target = _safe(rest)
    except ValueError as e:
        _bad(e)
    hidden = any(x.startswith(".") for x in target.relative_to(PAGES).parts)  # also through a symlink
    if hidden or not _in_share(share, target) or (target.is_dir() and not _real_index(target)):
        raise HTTPException(404, "not found")
    r = _serve(target, req, live=False, link=_share_link(share, token))
    r.headers["X-Robots-Tag"] = "noindex, nofollow"
    return r


def _origin(req: Request) -> str:
    """Base for the link: QILLQA_PUBLIC_URL, else the address the owner is using."""
    if os.environ.get("QILLQA_PUBLIC_URL"):
        return server.PUBLIC_URL
    proto = req.headers.get("x-forwarded-proto") or req.url.scheme
    host = req.headers.get("host") or req.url.netloc
    if host.endswith(".ts.net"):  # tailscale serve terminates TLS and forwards plain http
        proto = "https"
    return f"{proto}://{host}"


async def _json_obj(req: Request) -> dict:
    try:
        b = await req.json()
    except ValueError:
        b = None
    if not isinstance(b, dict):
        raise HTTPException(400, "expected a JSON object")
    return b


@app.post("/api/share")
async def api_share(req: Request):
    b = await _json_obj(req)
    hours, sub, label, path = b.get("hours", 168), b.get("subpages", False), b.get("label", ""), b.get("path")
    if (not isinstance(path, str) or not isinstance(sub, bool) or not isinstance(label, str)
            or isinstance(hours, bool) or not isinstance(hours, (int, float))):
        raise HTTPException(400, "expected path: string, hours: number, subpages: boolean, label: string")
    try:
        return server.share_create(path, hours, sub, label, _origin(req))
    except ValueError as e:
        _bad(e)


@app.get("/api/shares")
def api_shares(req: Request, path: str = ""):
    try:
        return server.share_list(path, _origin(req))
    except ValueError as e:
        _bad(e)


@app.post("/api/share/revoke")
async def api_share_revoke(req: Request):
    try:
        return server.share_revoke(str((await _json_obj(req)).get("id", "")))
    except ValueError as e:
        _bad(e)


CSS = """
:root{--bg:#f4f6fb;--fg:#1d2433;--mute:#5d6680;--line:#e2e6f0;--card:#fff;--acc:#5b3fd0;
--c1:#0f7c8c;--c2:#c2502e;--c3:#6d4bd1;--c4:#c98300;--c5:#2f8f5b;--ink:#000;--dim:0%;--side:#fbfcff;--zebra:#f7f8fe;
--code-bg:#fff0ea;--code-fg:#a8401c;--pre-bg:#f5f7ff;--pre-line:#e3e7fb;--quote-bg:#ecf8f6;--quote-fg:#174a52;
--shadow:#1d2433;color-scheme:light}
/* dark: gray-blue surfaces, full-contrast text */
:root[data-theme=dark]{--bg:#0f141d;--fg:#e4e8f1;--mute:#9aa3b8;--line:#2a3243;--card:#171d29;--acc:#a993ff;
--c1:#3ec3d3;--c2:#ff8f6b;--c3:#a68fff;--c4:#f0b84e;--c5:#5fcf8e;--ink:#fff;--dim:45%;--side:#131925;--zebra:#1b2230;
--code-bg:#2a1e1b;--code-fg:#ffad8f;--pre-bg:#121824;--pre-line:#2a3243;--quote-bg:#14262a;--quote-fg:#a7e3e6;
--shadow:#000;color-scheme:dark}
/* night: near-black, dimmer warm text, no cold blues (reading in the dark) */
:root[data-theme=night]{--bg:#0a0908;--fg:#d6c7ae;--mute:#8c7f6c;--line:#2a241d;--card:#13110e;--acc:#e0a35e;
--c1:#b9a15f;--c2:#d9805f;--c3:#c9955f;--c4:#d7b24f;--c5:#9dac67;--ink:#fff;--dim:55%;--side:#100e0c;--zebra:#18150f;
--code-bg:#231a12;--code-fg:#e8a77a;--pre-bg:#100e0b;--pre-line:#2a241d;--quote-bg:#1a1710;--quote-fg:#cdbb8e;
--shadow:#000;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,sans-serif}
a{color:var(--acc);text-underline-offset:3px}code,pre{font-family:ui-monospace,monospace;font-size:.9em}
article{max-width:880px;margin:24px auto;padding:36px 44px;background:var(--card);border-radius:18px;
box-shadow:0 1px 3px color-mix(in srgb,var(--shadow) 8%,transparent),0 12px 32px color-mix(in srgb,var(--shadow) 5%,transparent)}
article h1{font-size:2.3em;line-height:1.15;margin:0 0 .6em;padding-bottom:.35em;
background:linear-gradient(90deg,var(--c2),var(--c3) 55%,var(--c1));-webkit-background-clip:text;background-clip:text;color:transparent;
border-bottom:4px solid;border-image:linear-gradient(90deg,var(--c2),var(--c3),var(--c1)) 1}
article h2{--c:var(--c1);margin:1.8em 0 .7em;padding:8px 14px;border-left:6px solid var(--c);border-radius:8px;
background:color-mix(in srgb,var(--c) 11%,var(--card));color:color-mix(in srgb,var(--c) 75%,var(--ink))}
article h2:nth-of-type(5n+2){--c:var(--c2)}article h2:nth-of-type(5n+3){--c:var(--c3)}
article h2:nth-of-type(5n+4){--c:var(--c4)}article h2:nth-of-type(5n+5){--c:var(--c5)}
article h3{color:var(--c3);margin:1.6em 0 .5em;padding-bottom:4px;border-bottom:2px dotted color-mix(in srgb,var(--c3) 35%,var(--card))}
article li::marker{color:var(--c2)}
code{background:var(--code-bg);color:var(--code-fg);padding:1px 6px;border-radius:5px}
pre{background:var(--pre-bg);border:1px solid var(--pre-line);border-left:4px solid var(--c3);padding:14px 16px;overflow:auto;border-radius:10px}
pre code{background:none;color:var(--fg);padding:0}
blockquote{margin:1em 0;padding:10px 16px;background:var(--quote-bg);border-left:4px solid var(--c1);border-radius:8px;color:var(--quote-fg)}
blockquote p{margin:.4em 0}
table{border-collapse:separate;border-spacing:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:1em 0}
th{background:linear-gradient(90deg,color-mix(in srgb,var(--c1),#000 var(--dim)),color-mix(in srgb,var(--c3),#000 var(--dim)));color:#fff;text-align:left;font-weight:600}
.tw{overflow-x:auto;margin:1em 0}.tw table{margin:0}
td,th{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:var(--zebra)}
hr{border:0;height:3px;background:linear-gradient(90deg,var(--c2),var(--c3),var(--c1));border-radius:3px}
img,video{max-width:100%;border-radius:10px;box-shadow:0 4px 16px color-mix(in srgb,var(--shadow) 12%,transparent)}audio{width:100%}
@media (max-width:640px){article{margin:0;padding:22px 16px;border-radius:0;box-shadow:none}}
.login{max-width:340px;margin:18vh auto;display:grid;gap:12px;padding:28px;background:var(--card);border-radius:16px;
box-shadow:0 12px 32px color-mix(in srgb,var(--shadow) 10%,transparent)}
.login h1{margin:0;background:linear-gradient(90deg,var(--c2),var(--c3),var(--c1));-webkit-background-clip:text;background-clip:text;color:transparent}
input,button{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
::placeholder{color:var(--mute);opacity:1}
button{cursor:pointer;background:linear-gradient(90deg,color-mix(in srgb,var(--c3),#000 var(--dim)),color-mix(in srgb,var(--c1),#000 var(--dim)));color:#fff;border:0}.err{color:var(--c2);margin:0}
"""
THEMES = ("light", "dark", "night")
THEME_COOKIE = "qillqa_theme"


def _theme(req: Request | None) -> str:
    """The theme cookie, whitelisted: it goes into an HTML attribute."""
    t = req.cookies.get(THEME_COOKIE, "") if req else ""
    return t if t in THEMES else "light"


def _shell(title: str, body: str, theme: str = "light") -> str:
    theme = theme if theme in THEMES else "light"
    return (f'<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>")


INDEX = """
<header><button id="side" type="button" title="Show / hide the sidebar" aria-label="Show / hide the sidebar">☰</button><a href="#" class="brand">qillqa</a>
<input id="q" type="search" placeholder="Search by meaning…" autocomplete="off">
<label id="scope" hidden title="Search only this folder and its subpages"><input type="checkbox" id="only"> only in <b id="scopename"></b></label>
<div id="me" class="out"><button id="avatar" type="button" aria-expanded="false" aria-controls="settings" title="Settings" aria-label="Settings"><svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" fill="currentColor"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7z"/></svg></button>
<div id="settings" hidden><fieldset><legend>Theme</legend>
<label><input type="radio" name="theme" value="light"> ☀️ White</label>
<label><input type="radio" name="theme" value="dark"> 🌙 Dark</label>
<label><input type="radio" name="theme" value="night"> 🌌 Night</label></fieldset>
<fieldset><legend>Header</legend><label><input type="checkbox" id="autohide"> Auto-hide</label></fieldset>
<a href="/logout">Sign out</a></div></div></header>
<div id="peek" title="Show the header"></div>
<button id="menu" type="button" aria-label="Menu" title="Menu">☰</button>
<main><aside><button id="toggle">Pages</button><nav id="tree"></nav></aside>
<section><div id="results" hidden></div><div id="crumb"></div>
<iframe id="page" sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals allow-top-navigation-by-user-activation" allow="fullscreen" title="page"></iframe>
<div id="edit" hidden><div class="ed-bar"><b id="ed-name"></b><small id="ed-msg"></small>
<button id="ed-cancel" type="button" class="ghost">Close</button><button id="ed-save" type="button">Save</button></div>
<textarea id="ed-text" spellcheck="false" aria-label="Page source"></textarea>
<details id="ed-files" open><summary>Files in this page</summary><ul id="ed-list"></ul></details></div>
<div id="cards" hidden><div class="cards-head"><h2 id="cards-title"></h2><label>Sort <select id="sort"><option value="date">Last changed</option><option value="name">Name</option></select></label></div><div id="cards-body"></div></div></section></main>
<dialog id="share"><form method="dialog">
<h3>Share <span id="sh-title"></span></h3>
<p class="sh-note">Anyone with the link can open this page, read-only and without signing in, until it expires or you revoke it.</p>
<label>Expires in <select id="sh-hours"><option value="1">1 hour</option><option value="24">24 hours</option><option value="168" selected>7 days</option><option value="720">30 days</option></select></label>
<label id="sh-sub"><input type="checkbox" id="sh-subpages"> Include its subpages</label>
<label>Label <input id="sh-label" maxlength="120" placeholder="Who is it for? (optional)"></label>
<div class="sh-row"><button type="button" id="sh-create">Create link</button><button value="close" class="ghost">Close</button></div>
<div id="sh-out" hidden><input id="sh-url" readonly aria-label="Share link"><button type="button" id="sh-copy">Copy</button></div>
<small id="sh-msg"></small>
<h4>Links of this page</h4><ul id="sh-list"></ul></form></dialog>
<style>
header{display:flex;gap:12px;align-items:center;padding:10px 16px;background:var(--card);
border-bottom:3px solid;border-image:linear-gradient(90deg,var(--c2),var(--c3),var(--c1)) 1}
.brand{font-weight:800;font-size:1.15em;text-decoration:none;background:linear-gradient(90deg,var(--c2),var(--c3));
-webkit-background-clip:text;background-clip:text;color:transparent}
#q{flex:1;min-width:0;background:var(--bg)}#q:focus{outline:2px solid var(--c3);background:var(--card)}
#me{position:relative;flex:none}#avatar{display:flex;align-items:center;justify-content:center;width:36px;height:36px;padding:0;border-radius:50%}
#avatar:focus-visible{outline:2px solid var(--c3);outline-offset:2px}
#settings{position:absolute;right:0;top:calc(100% + 8px);z-index:20;min-width:190px;padding:10px;background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:12px;box-shadow:0 8px 24px color-mix(in srgb,var(--shadow) 22%,transparent)}
#settings fieldset{border:0;margin:0;padding:0;display:grid;gap:2px}
#settings fieldset+fieldset{margin-top:6px;padding-top:6px;border-top:1px solid var(--line)}
#settings legend{padding:0 8px 4px;font-size:.75em;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
#settings label{display:flex;gap:10px;align-items:center;padding:6px 8px;border-radius:8px;cursor:pointer}
#settings label:hover{background:color-mix(in srgb,var(--c3) 10%,var(--card))}
#settings label:has(:checked){background:color-mix(in srgb,var(--c3) 16%,var(--card));font-weight:600}
#settings input{margin:0;padding:0;accent-color:var(--c3)}
#settings a{display:block;margin-top:6px;padding:8px;border-top:1px solid var(--line);color:var(--fg);text-decoration:none}
#settings a:hover{color:var(--c2)}
#scope{font-size:.85em;color:var(--mute);white-space:nowrap;max-width:30vw;overflow:hidden;text-overflow:ellipsis;cursor:pointer}
#scope b{color:var(--c1)}#scope input{accent-color:var(--c3);vertical-align:middle}
main{display:grid;grid-template-columns:280px 1fr;height:calc(100vh - 62px)}
aside{background:var(--side);border-right:1px solid var(--line);overflow:auto;padding:10px}#toggle{display:none}
nav ul{list-style:none;margin:0;padding-left:14px}nav>ul{padding-left:0}
#side{background:none;border:0;color:var(--mute);font-size:1.2em;padding:2px 6px;cursor:pointer}#side:hover{color:var(--c3)}
#q{height:46px}header:not(.searching) #scope{display:none}
/* desktop auto-hide: the header goes away while reading; a thin strip at the top brings it back */
#peek{display:none}
@media (min-width:761px){body.autohide.hid header{display:none}body.autohide.hid main{height:100vh}
body.autohide.hid #peek{display:block;position:fixed;top:0;left:0;right:0;height:8px;z-index:30;cursor:pointer;
background:linear-gradient(90deg,var(--c2),var(--c3),var(--c1));opacity:.45}body.autohide.hid #peek:hover{opacity:1}}
body.noside main{grid-template-columns:1fr}body.noside aside{display:none}
.row{display:flex;align-items:center}.fold,.leaf{flex:none;width:20px}
.fold{background:none;border:0;padding:0;color:var(--mute);cursor:pointer;line-height:1;transition:transform .15s}
.fold:hover{color:var(--c3)}li.open>.row>.fold{transform:rotate(90deg)}nav li>ul{display:none}nav li.open>ul{display:block}
nav a{flex:1;min-width:0;display:block;padding:4px 8px;border-radius:6px;color:var(--fg);text-decoration:none;overflow-wrap:anywhere}
nav a:hover{background:color-mix(in srgb,var(--c3) 8%,var(--card))}
nav a.sec{color:var(--c1);font-weight:600}
nav a.in{color:var(--c3)}
nav a.on{background:color-mix(in srgb,var(--c3) 14%,var(--card));color:color-mix(in srgb,var(--c3) 80%,var(--ink));font-weight:600;
box-shadow:inset 3px 0 0 var(--c3)}
section{display:flex;flex-direction:column;min-width:0}iframe{flex:1;border:0;width:100%;background:var(--bg)}
#crumb{padding:6px 16px;color:var(--mute);font-size:.85em;font-weight:600;overflow-wrap:anywhere}
#crumb .exps{float:right;display:flex;gap:12px;color:inherit}#crumb a.exp,#crumb a.edl{font-weight:500;white-space:nowrap}
#crumb a{color:var(--c1);text-decoration:none;border-radius:4px}#crumb a:hover,#crumb a:focus-visible{color:var(--c3);text-decoration:underline}#crumb span{color:var(--fg)}
/* root (logo) and folders without a page: one card per project, last changed first */
#cards{flex:1;overflow:auto;padding:4px 16px 24px}
.cards-head{display:flex;align-items:center;gap:12px;margin:8px 0 14px}.cards-head h2{margin:0;font-size:1.3em;flex:1}
.cards-head label{color:var(--mute);font-size:.9em}.cards-head select{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.card{--c:var(--c1);display:flex;flex-direction:column;gap:6px;padding:14px 16px;border-radius:14px;background:var(--card);border:1px solid var(--line);
border-top:4px solid var(--c);color:var(--fg);text-decoration:none;box-shadow:0 1px 3px color-mix(in srgb,var(--shadow) 8%,transparent)}
.card:nth-child(5n+2){--c:var(--c2)}.card:nth-child(5n+3){--c:var(--c3)}.card:nth-child(5n+4){--c:var(--c4)}.card:nth-child(5n+5){--c:var(--c5)}
.card:hover,.card:focus-visible{border-color:var(--c);outline:0;box-shadow:0 6px 18px color-mix(in srgb,var(--shadow) 14%,transparent)}
.card b{font-size:1.05em;overflow-wrap:anywhere}.card small{color:var(--mute)}.card .last{color:var(--c);font-size:.9em;overflow-wrap:anywhere}
.none{color:var(--mute)}
#results{padding:8px 16px;background:var(--card);border-bottom:1px solid var(--line);max-height:45vh;overflow:auto}
.hit{display:block;padding:10px 12px;margin:6px 0;border-radius:10px;border:1px solid var(--line);color:var(--fg);text-decoration:none}
.hit:hover{border-color:var(--c3);background:color-mix(in srgb,var(--c3) 5%,var(--card))}
.hit b{color:var(--c3)}.hit small{color:var(--c1);display:block}
@media (max-width:760px){header{flex-wrap:wrap}#q{order:2;flex-basis:100%}.out{margin-left:auto}#scope{order:3;max-width:100%}
main{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid var(--line)}
#side{display:none}#toggle{display:block;width:100%;background:var(--card);color:var(--fg);border:1px solid var(--line)}
nav{display:none}nav.open{display:block}iframe{height:75vh;flex:none}}
/* editor: the page source and the files in its folder */
#edit{flex:1;display:flex;flex-direction:column;gap:8px;padding:4px 16px 16px;min-height:0}#edit[hidden]{display:none}
.ed-bar{display:flex;align-items:center;gap:10px}.ed-bar b{overflow-wrap:anywhere}.ed-bar small{flex:1;color:var(--mute)}
.ed-bar small.bad{color:var(--c2)}.ed-bar button{padding:6px 14px}button.ghost{background:var(--card);color:var(--fg);border:1px solid var(--line)}
#share{border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--fg);padding:18px 20px;width:min(520px,calc(100vw - 32px))}
#share::backdrop{background:color-mix(in srgb,var(--shadow) 45%,transparent)}
#share [hidden]{display:none!important}#share form{display:grid;gap:10px}#share h3{margin:0}#share h4{margin:6px 0 0;font-size:.95em}
#share label{display:grid;gap:4px;font-size:.9em;color:var(--mute)}#share label#sh-sub{display:flex;gap:8px;align-items:center}
#share select,#share input{font:inherit;padding:7px 9px}.sh-note{margin:0;color:var(--mute);font-size:.85em}
.sh-row,#sh-out{display:flex;gap:8px;flex-wrap:wrap}#sh-out input{flex:1;min-width:0}#sh-msg{color:var(--mute)}#sh-msg.bad{color:var(--c2)}
#sh-list{list-style:none;margin:0;padding:0;display:grid;gap:6px;max-height:40vh;overflow:auto}
#sh-list li{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:7px 9px;border:1px solid var(--line);border-radius:9px;font-size:.88em}
#sh-list li .who{flex:1;min-width:140px}#sh-list li small{color:var(--mute)}#sh-list li.off{opacity:.55}
#sh-list button{padding:4px 10px;font-size:.9em}
#ed-text{flex:1;min-height:40vh;width:100%;resize:vertical;padding:12px 14px;border:1px solid var(--line);border-radius:10px;
background:var(--card);color:var(--fg);font:14px/1.55 ui-monospace,monospace;tab-size:2}#ed-text:focus{outline:2px solid var(--c3)}
#ed-files{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px;max-height:30vh;overflow:auto}
#ed-files summary{cursor:pointer;font-weight:600;color:var(--c1)}#ed-list{list-style:none;margin:6px 0 0;padding:0}
#ed-list li{display:flex;gap:12px;padding:4px 0;border-top:1px solid var(--line)}#ed-list a{flex:1;overflow-wrap:anywhere}
#ed-list small{color:var(--mute);white-space:nowrap}
#menu{display:none}
/* phones: only the page; a small floating ☰ brings back header + pages */
@media (max-width:760px){#menu{display:flex;align-items:center;justify-content:center;position:fixed;right:12px;bottom:12px;z-index:10;width:44px;height:44px;padding:0;border-radius:50%;font-size:1.2em;color:var(--fg);background:var(--card);border:1px solid var(--line);box-shadow:0 2px 10px #0003;opacity:.85}
body:not(.bar) header,body:not(.bar) aside,body:not(.bar) #crumb,body:not(.bar) #results{display:none}
header:not(.searching) #scope:not([hidden]){display:inline}#toggle{display:none}nav{display:block}aside{max-height:50vh}iframe{height:100vh;height:100dvh}}
</style>
<script>
const $ = s => document.querySelector(s);
function mk(tag, props, ...kids){const e=Object.assign(document.createElement(tag),props);e.append(...kids);return e}
async function api(u){const r=await fetch(u);if(r.status===401)location='/login';return r.json()}
function load(k,d){try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}}
function save(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch{}}
// folders start collapsed; the ones the user opens are remembered
const opened=new Set(load('qillqa.open',[]));
function renderTree(nodes){
  const ul=mk('ul');
  for(const n of nodes){
    const a=mk('a',{href:'#'+n.path,className:n.page?'':'sec',textContent:n.title});a.dataset.path=n.path;
    const li=mk('li');
    if(n.children.length){
      const b=mk('button',{type:'button',className:'fold',textContent:'▸',title:'Expand / collapse'});
      const set=on=>{li.classList.toggle('open',on);b.setAttribute('aria-expanded',on)};set(opened.has(n.path));
      b.onclick=()=>{const on=!li.classList.contains('open');set(on);
        on?opened.add(n.path):opened.delete(n.path);save('qillqa.open',[...opened])};
      li.append(mk('div',{className:'row'},b,a),renderTree(n.children));
    }else li.append(mk('div',{className:'row'},mk('span',{className:'leaf'}),a));
    ul.append(li);
  }
  return ul;
}
let nodes=[],treeAt=0;
async function loadTree(){nodes=await api('/api/tree');treeAt=Date.now();$('#tree').replaceChildren(renderTree(nodes));mark()}
// cards: a subtree's newest change, its page count and the page that changed last
function stats(n){let s={last:n.mtime,lastNode:n.mtime?n:null,pages:n.page?1:0};
  for(const c of n.children){const k=stats(c);s.pages+=k.pages;if(k.last>s.last){s.last=k.last;s.lastNode=k.lastNode}}return s}
const nodeAt=(list,p)=>{for(const n of list){if(n.path===p)return n;if(p.startsWith(n.path+'/')){const f=nodeAt(n.children,p);if(f)return f}}return null};
const ago=new Intl.RelativeTimeFormat(undefined,{numeric:'auto'});
function when(t){if(!t)return'never';const d=(t*1000-Date.now())/1000,a=Math.abs(d);
  const [v,u]=a<3600?[d/60,'minute']:a<86400?[d/3600,'hour']:a<86400*30?[d/86400,'day']:a<86400*365?[d/86400/30,'month']:[d/86400/365,'year'];
  return ago.format(Math.round(v),u)}
function card(n){const s=stats(n),a=mk('a',{className:'card',href:'#'+n.path},mk('b',{textContent:n.title}),
    ...(n.path.toLowerCase()!==n.title.toLowerCase()?[mk('small',{textContent:n.path})]:[]),
    mk('small',{textContent:'Changed '+when(s.last)+' · '+s.pages+(s.pages===1?' page':' pages'),
      title:s.last?new Date(s.last*1000).toLocaleString():''}));
  if(s.lastNode&&s.lastNode!==n)a.append(mk('span',{className:'last',textContent:'Last: '+s.lastNode.title}));
  if(s.lastNode&&s.lastNode!==n)a.onclick=e=>{if(e.target.classList.contains('last')){e.preventDefault();location.hash=s.lastNode.path}};
  return a}
async function cards(p){
  if(Date.now()-treeAt>5000)await loadTree();  // fresh dates when coming back to the list
  const list=p?(nodeAt(nodes,p)||{children:[]}).children:nodes,by=$('#sort').value,body=$('#cards-body');
  $('#cards-title').textContent=p?(nodeAt(nodes,p)||{title:p}).title:'Projects';
  if(!list.length){body.replaceChildren(mk('p',{className:'none',textContent:p?'Nothing here yet.':'No pages yet.'}));return}
  const rows=list.map(n=>[n,stats(n).last]);
  rows.sort(by==='name'?(a,b)=>a[0].title.localeCompare(b[0].title):(a,b)=>b[1]-a[1]);
  body.replaceChildren(mk('div',{className:'grid'},...rows.map(r=>card(r[0]))))}
$('#sort').value=load('qillqa.sort','date');
$('#sort').onchange=()=>{save('qillqa.sort',$('#sort').value);cards(decodeURIComponent(location.hash.slice(1)))};
function mark(){const p=decodeURIComponent(location.hash.slice(1));
  document.querySelectorAll('nav a').forEach(a=>{a.classList.toggle('on',a.dataset.path===p);
    a.classList.toggle('in',!!p&&p.startsWith(a.dataset.path+'/'))})}  // 'in': holds the open page
// breadcrumb: Projects / a / b / page — every parent is a link
function crumb(p){const parts=p?p.split('/'):[],out=parts.length?[mk('a',{href:'#',textContent:'Projects'})]:[];
  parts.forEach((s,i)=>out.push(' / ',i<parts.length-1?mk('a',{href:'#'+parts.slice(0,i+1).join('/'),textContent:s}):mk('span',{textContent:s})));
  // export: one .html that opens on any device, offline (a folder with no page exports what is under it)
  const n=p&&nodeAt(nodes,p),ex=(t,sub)=>mk('a',{className:'exp',download:'',textContent:t,
    href:'/api/export?path='+encodeURIComponent(p)+(sub?'&subpages=1':''),title:'Download one .html file that opens on any device, even offline'});
  const below=x=>x.children.some(c=>c.page||below(c));  // a real page below (an assets/ folder is not one)
  const ed=n&&n.page?[mk('a',{className:'edl',href:'#'+p,textContent:editing?'👁 View':'✎ Edit',
    title:'Edit the page source and see the files in its folder',onclick:e=>{e.preventDefault();editing?closeEdit():openEdit(p)}})]:[];
  const sh=n&&n.page?[mk('a',{className:'edl',href:'#'+p,textContent:'🔗 Share',
    title:'Create a temporary read-only link to this page',onclick:e=>{e.preventDefault();openShare(p)}})]:[];
  if(n)out.push(mk('span',{className:'exps'},...ed,...sh,ex('⤓ Export',0),...(n.page&&below(n)?[ex('⤓ + subpages',1)]:[])));
  $('#crumb').replaceChildren(...out)}
// editor: raw index.md/html (saved with its version: 409 if the MCP or another tab changed it)
let editing='',edVer='',edSaved='',edSeq=0,saving=-1;  // edSeq: a reply for an older open/save is ignored
const dirty=()=>!!editing&&$('#ed-text').value!==edSaved;
const size=b=>b<1024?b+' B':b<1048576?(b/1024).toFixed(1)+' KB':(b/1048576).toFixed(1)+' MB';
const edMsg=(t,bad)=>{$('#ed-msg').textContent=t;$('#ed-msg').classList.toggle('bad',!!bad)};
const here=()=>decodeURIComponent(location.hash.slice(1));
async function openEdit(p){
  const my=++edSeq,r=await fetch('/api/source?path='+encodeURIComponent(p));
  if(r.status===401){location='/login';return}
  const d=await r.json().catch(()=>({}));
  if(my!==edSeq||here()!==p)return;  // clicked again, closed or navigated meanwhile
  if(!r.ok){$('#crumb').textContent='⚠ '+(d.detail||r.status);return}
  const t=$('#ed-text');editing=p;edVer=d.version;t.value=d.content;edSaved=t.value;  // the textarea's own copy
  t.readOnly=!!d.readonly;$('#ed-save').disabled=!!d.readonly;
  $('#ed-name').textContent='index.'+d.format;edMsg(d.readonly?'⚠ Read only ('+d.readonly+'): edit it where it lives.':'',!!d.readonly);
  $('#ed-list').replaceChildren(...(d.files.length?d.files.map(f=>mk('li',{},
    mk('a',{href:f.url,target:'_blank',rel:'noopener',textContent:f.name}),
    mk('small',{textContent:size(f.size)+' · '+when(f.mtime),title:new Date(f.mtime*1000).toLocaleString()}))):
    [mk('li',{className:'none',textContent:'No files: only the page itself.'})]));
  $('#ed-files summary').textContent='Files in this page ('+d.files.length+')';
  $('#page').hidden=true;$('#edit').hidden=false;crumb(p);t.setSelectionRange(0,0);t.focus();t.scrollTop=0}
function closeEdit(force){
  if(!force&&dirty()&&!confirm('Discard your unsaved changes?'))return false;
  const p=editing;editing='';edSeq++;$('#edit').hidden=true;
  if(p&&p===here()){$('#page').hidden=false;$('#page').src='/view?path='+encodeURIComponent(p)+'&r='+Date.now();crumb(p)}
  return true}
async function saveEdit(){
  const p=editing,text=$('#ed-text').value,my=edSeq;if(!p||saving===my||$('#ed-save').disabled)return;
  saving=my;edMsg('Saving…');  // one save in flight per editor session
  try{
    const r=await fetch('/api/source',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:p,content:text,version:edVer})});
    if(r.status===401){location='/login';return}
    const d=await r.json().catch(()=>({}));
    if(my!==edSeq||editing!==p)return;  // that editor is gone: its reply must not touch this one
    if(r.status===409){edMsg('⚠ The page changed since you opened it. Copy your text, then Close and Edit again.',1);return}
    if(!r.ok){edMsg('⚠ '+(d.detail||r.status),1);return}
    edVer=d.version;edSaved=text;edMsg('Saved '+new Date().toLocaleTimeString())
  }catch(e){if(my===edSeq)edMsg('⚠ Not saved: '+e.message,1)}
  finally{if(saving===my)saving=-1}}
$('#ed-save').onclick=saveEdit;$('#ed-cancel').onclick=()=>closeEdit();
$('#ed-text').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();saveEdit()}});
addEventListener('beforeunload',e=>{if(dirty())e.preventDefault()});
let scope='';
async function show(){
  const p=decodeURIComponent(location.hash.slice(1));
  if(editing&&p!==editing&&!closeEdit()){location.hash=editing;return}  // kept the unsaved text
  mark();
  if(!scope){$('#scopename').textContent=p;$('#scope').hidden=!p}
  crumb(p);
  // the root (logo) and a folder with no page of its own show cards; no iframe (its 404 would be unthemed JSON)
  const sec=!!p&&!!document.querySelector('nav a.sec[data-path="'+CSS.escape(p)+'"]');
  if(!p||sec){$('#page').hidden=true;$('#cards').hidden=false;$('#page').src='about:blank';$('nav').classList.remove('open');return cards(p)}
  if(editing)return;
  $('#cards').hidden=true;$('#page').hidden=false;
  $('#page').src='/view?path='+encodeURIComponent(p);$('nav').classList.remove('open');
}
let t;
// the search box is always there; 'searching' (box in use) only shows the folder filter
const head=$('header');
$('#q').addEventListener('focus',()=>head.classList.add('searching'));
head.addEventListener('focusout',e=>{if(!head.contains(e.relatedTarget)&&!$('#q').value.trim())head.classList.remove('searching')});
$('#q').addEventListener('keydown',e=>{if(e.key==='Escape'){$('#q').value='';find();$('#q').blur();head.classList.remove('searching')}});
addEventListener('keydown',e=>{if(e.key==='/'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();reveal();$('#q').focus()}});
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
document.body.classList.toggle('noside',load('qillqa.noside',false));
$('#side').onclick=()=>save('qillqa.noside',document.body.classList.toggle('noside'));
// checklist pages post {qillqa: check|move} here; only the shown page, only its own iframe
addEventListener('message',async e=>{
  const f=$('#page'),d=e.data,p=decodeURIComponent(location.hash.slice(1));
  if(e.source!==f.contentWindow||!d||!['check','move'].includes(d.qillqa)||!p)return;
  const r=await fetch('/api/page/edit',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:p,op:d.qillqa,i:d.i,to:d.to,line:d.line,after:!!d.after,cap:d.cap})});
  if(r.status===401){location='/login';return}
  if(decodeURIComponent(location.hash.slice(1))!==p)return;  // navigated away meanwhile
  if(!r.ok&&r.status!==409)$('#crumb').textContent='⚠ '+((await r.json()).detail||r.status);
  f.src='/view?path='+encodeURIComponent(p)+'&r='+Date.now()+'#qy='+(d.y|0);
});
// avatar -> settings (theme, sign out). The theme is a cookie: the server renders every page in
// it, including the sandboxed page iframe, which can't read our storage or hear us.
const me=$('#me'),pane=$('#settings'),av=$('#avatar');
const settings=on=>{pane.hidden=!on;av.setAttribute('aria-expanded',on);if(!on)hide()};
av.onclick=()=>settings(pane.hidden);
addEventListener('click',e=>{if(!me.contains(e.target))settings(false)});
addEventListener('blur',()=>settings(false));  // a click into the page iframe
me.addEventListener('focusout',e=>{if(e.relatedTarget&&!me.contains(e.relatedTarget))settings(false)});
addEventListener('keydown',e=>{if(e.key==='Escape'&&!pane.hidden){settings(false);if(me.contains(document.activeElement))av.focus()}});
document.querySelectorAll('[name=theme]').forEach(r=>{
  r.checked=r.value===document.documentElement.dataset.theme;
  r.onchange=()=>{document.cookie='qillqa_theme='+r.value+';path=/;max-age=31536000;samesite=lax';
    document.documentElement.dataset.theme=r.value;
    const p=decodeURIComponent(location.hash.slice(1));
    if(!$('#page').hidden)$('#page').src='/view?path='+encodeURIComponent(p)+'&r='+Date.now()}});
// header auto-hide (desktop, on by default): it stays while the mouse is on it or it is in use
const auto=$('#autohide');let hideT;
const busy=()=>head.matches(':hover')||head.contains(document.activeElement)||!!$('#q').value.trim()||!pane.hidden;
function hide(ms=800){clearTimeout(hideT);hideT=setTimeout(()=>{if(!busy())document.body.classList.add('hid')},ms)}
function reveal(){clearTimeout(hideT);document.body.classList.remove('hid')}
auto.checked=load('qillqa.autohide',true);document.body.classList.toggle('autohide',auto.checked);
auto.onchange=()=>{save('qillqa.autohide',auto.checked);document.body.classList.toggle('autohide',auto.checked)};
$('#peek').addEventListener('mouseenter',reveal);$('#peek').addEventListener('click',reveal);
head.addEventListener('mouseleave',()=>hide());
head.addEventListener('focusout',e=>{if(!head.contains(e.relatedTarget))hide()});
hide(1500);  // seen at load, then out of the way
const bar=on=>{document.body.classList.toggle('bar',on);$('#menu').textContent=on?'✕':'☰';settings(false)};
$('#menu').onclick=()=>bar(!document.body.classList.contains('bar'));
addEventListener('hashchange',()=>bar(false));  // picked a page: back to reading
$('#tree').addEventListener('click',e=>{if(e.target.closest('a'))bar(false)});  // ...also the one already open
// share links: temporary read-only URL of one page (/s/<token>), revocable
let shareP='',shGen=0;
const hasSub=x=>x.children.some(c=>c.page||hasSub(c));
const shMsg=t=>{t=t||'';$('#sh-msg').textContent=t;$('#sh-msg').classList.toggle('bad',t.startsWith('⚠'))};
function openShare(p){const n=nodeAt(nodes,p);shareP=p;shGen++;$('#sh-list').replaceChildren();$('#sh-url').value='';$('#sh-title').textContent=(n&&n.title)||p;
  $('#sh-sub').hidden=!(n&&hasSub(n));$('#sh-subpages').checked=false;$('#sh-label').value='';$('#sh-out').hidden=true;shMsg('');
  $('#share').showModal();listShares()}
async function copyLink(url){const g=shGen;try{await navigator.clipboard.writeText(url);if(g===shGen)shMsg('Link copied')}
  catch(_){if(g!==shGen)return;$('#sh-out').hidden=false;$('#sh-url').value=url;$('#sh-url').select();shMsg('Copy it with Ctrl+C')}}
async function listShares(){const g=shGen,r=await fetch('/api/shares?path='+encodeURIComponent(shareP));if(g!==shGen)return;
  const rows=r.ok?await r.json():[];if(g!==shGen)return;const when=t=>new Date(t*1000).toLocaleString([],{dateStyle:'medium',timeStyle:'short'});
  $('#sh-list').replaceChildren(...(rows.length?rows.map(s=>{const on=s.state==='active';
    return mk('li',{className:on?'':'off'},mk('span',{className:'who'},mk('b',{textContent:s.label||'(no label)'}),mk('br'),
      mk('small',{textContent:(on?'expires '+when(s.expires):s.state)+(s.subpages?' · + subpages':'')+' · opened '+s.uses+'×'})),
      ...(on?[mk('button',{type:'button',className:'ghost',textContent:'Copy',onclick:()=>copyLink(s.url)}),
             mk('button',{type:'button',className:'ghost',textContent:'Revoke',onclick:()=>revokeShare(s.id)})]:[]))})
    :[mk('li',{className:'off',textContent:'No links yet.'})]))}
async function revokeShare(id){const g=shGen,r=await fetch('/api/share/revoke',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  if(g!==shGen)return;shMsg(r.ok?'Link revoked':'⚠ '+r.status);listShares()}
$('#sh-create').onclick=async()=>{shMsg('');const g=shGen;
  const r=await fetch('/api/share',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:shareP,
    hours:+$('#sh-hours').value,subpages:!$('#sh-sub').hidden&&$('#sh-subpages').checked,label:$('#sh-label').value})});
  const d=await r.json();if(g!==shGen)return;if(!r.ok){shMsg('⚠ '+(d.detail||r.status));return}
  $('#sh-out').hidden=false;$('#sh-url').value=d.url;$('#sh-url').select();copyLink(d.url);listShares()};
$('#sh-copy').onclick=()=>copyLink($('#sh-url').value);
$('#share').addEventListener('close',()=>{shGen++});  // a reply that lands after closing changes nothing
addEventListener('hashchange',show);loadTree().then(show);
</script>"""


@app.get("/", response_class=HTMLResponse)
def index(req: Request):
    return _shell("qillqa", INDEX, _theme(req))
