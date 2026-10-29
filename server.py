#!/usr/bin/env python3
"""qillqa MCP (stdio): a local notebook of nested pages.

A page is a folder under QILLQA_PAGES_DIR holding index.md or index.html; its
subfolders are subpages. Writes go straight to disk (no server needed), then
the local web server (webui.py) is asked to index in the background; the MCP
starts it on demand (~0.5 s).

Do not import torch/numpy/sentence_transformers here: this module loads during
the MCP handshake and webui.py imports it.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

HOST = os.environ.get("QILLQA_HOST", "127.0.0.1")
PORT = int(os.environ.get("QILLQA_PORT", "47391"))
TOKEN = os.environ.get("QILLQA_TOKEN", "")
MODEL = os.environ.get("QILLQA_MODEL", "Qwen/Qwen3-Embedding-0.6B")
PAGES = (BASE / os.path.expanduser(os.environ.get("QILLQA_PAGES_DIR", "pages"))).resolve()
VAR = BASE / "var"
INDEX_NAMES = ("index.md", "index.html")
URL = f"http://{HOST}:{PORT}"

mcp = FastMCP("qillqa")


# ---------------------------------------------------------------- pages

def _safe(path: str) -> Path:
    """Page path -> folder inside PAGES. Rejects anything that escapes it."""
    p = (PAGES / (path or "").strip().strip("/")).resolve()
    if p != PAGES and PAGES not in p.parents:
        raise ValueError(f"path outside the notebook: {path!r}")
    return p


def _rel(d: Path) -> str:
    return "" if d == PAGES else d.relative_to(PAGES).as_posix()


def _index(d: Path) -> Path | None:
    return next((d / n for n in INDEX_NAMES if (d / n).is_file()), None)


def _title(d: Path) -> str:
    idx = _index(d)
    if idx:
        text = idx.read_text(errors="replace")
        m = (re.search(r"^# +(.+)$", text, re.M) if idx.suffix == ".md"
             else re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S))
        if m and m.group(1).strip():
            return m.group(1).strip()
    return d.name or "qillqa"


def _children(d: Path) -> list[Path]:
    return sorted((c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")),
                  key=lambda c: c.name.lower()) if d.is_dir() else []


def tree(path: str = "") -> list[dict]:
    def node(d: Path) -> dict:
        return {"path": _rel(d), "title": _title(d), "page": _index(d) is not None,
                "children": [node(c) for c in _children(d)]}
    root = _safe(path)
    if not root.is_dir():
        raise ValueError(f"does not exist: {path!r}")
    return [node(c) for c in _children(root)]


def read_page(path: str) -> dict:
    d = _safe(path)
    idx = _index(d)
    if not idx:
        raise ValueError(f"no page at {path!r}")
    return {"path": _rel(d), "title": _title(d), "format": idx.suffix[1:],
            "content": idx.read_text(errors="replace"),
            "children": [_rel(c) for c in _children(d)], "files": _files(d),
            "url": f"{URL}/#{_rel(d)}"}


def _files(d: Path) -> list[str]:
    return sorted(f.name for f in d.iterdir()
                  if f.is_file() and f.name not in INDEX_NAMES and not f.name.startswith("."))


_MEDIA = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif",
          ".mp4", ".webm", ".mov", ".m4v", ".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def attach(path: str, source: str, name: str = "") -> dict:
    """Copy a local file into the page folder."""
    d = _safe(path)
    if d == PAGES or not d.is_dir():
        raise ValueError(f"page {path!r} does not exist; create it first")
    src = Path(os.path.expanduser(source))
    if not src.is_file():
        raise ValueError(f"file {source!r} does not exist")
    name = name or src.name
    if "/" in name or "\\" in name or name.startswith(".") or name in INDEX_NAMES:
        raise ValueError(f"invalid file name: {name!r}")
    shutil.copyfile(src, d / name)
    ref = urllib.parse.quote(name)
    md = f"![{name}]({ref})" if Path(name).suffix.lower() in _MEDIA else f"[{name}]({ref})"
    return {"page": _rel(d), "file": name, "markdown": md}


@contextlib.contextmanager
def page_lock():
    """One page writer at a time across processes: this MCP server and the web UI's
    checklist edits read-check-write under it, so neither loses the other's write."""
    VAR.mkdir(parents=True, exist_ok=True)
    with open(VAR / "write.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _locked(fn):
    @functools.wraps(fn)
    def run(*a, **k):
        with page_lock():
            return fn(*a, **k)
    return run


@_locked
def write_page(path: str, content: str, fmt: str, create: bool) -> dict:
    if fmt not in ("md", "html"):
        raise ValueError("format must be 'md' or 'html'")
    d = _safe(path)
    if d == PAGES:
        raise ValueError("give a page path, e.g. 'project/note'")
    exists = _index(d) is not None
    if create and exists:
        raise ValueError(f"{path!r} already exists; use update_page")
    if not create and not exists:
        raise ValueError(f"{path!r} does not exist; use create_page")
    d.mkdir(parents=True, exist_ok=True)
    for n in INDEX_NAMES:  # switching format must not leave the old index behind
        if n != f"index.{fmt}":
            (d / n).unlink(missing_ok=True)
    (d / f"index.{fmt}").write_text(content)
    return {"path": _rel(d), "title": _title(d), "url": f"{URL}/#{_rel(d)}"}


@_locked
def _move(src: str, dst: str) -> dict:
    s, t = _safe(src), _safe(dst)
    if s == PAGES or not s.is_dir():
        raise ValueError(f"{src!r} does not exist")
    if t.exists():
        raise ValueError(f"{dst!r} already exists")
    if t == s or s in t.parents:
        raise ValueError("cannot move a page inside itself")
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(t))
    return {"path": _rel(t), "url": f"{URL}/#{_rel(t)}"}


@_locked
def _delete(path: str, recursive: bool = False) -> dict:
    d = _safe(path)
    if d == PAGES or not d.is_dir():
        raise ValueError(f"{path!r} does not exist")
    kids = _children(d)
    if kids and not recursive:
        raise ValueError(f"{path!r} has {len(kids)} subpages; pass recursive=True")
    shutil.rmtree(d)
    return {"deleted": _rel(d)}


# ---------------------------------------------------------------- server

def _health() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=1) as r:
            return json.load(r).get("app") == "qillqa"
    except Exception:
        return False


def start() -> dict:
    if _health():
        return {"running": True, "url": URL}
    if not TOKEN:
        raise RuntimeError("QILLQA_TOKEN is empty in .env; run setup.sh or set a token")
    VAR.mkdir(exist_ok=True)
    log = open(VAR / "server.log", "ab")
    proc = subprocess.Popen(
        [str(BASE / ".venv/bin/uvicorn"), "webui:app", "--host", HOST, "--port", str(PORT)],
        cwd=BASE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    (VAR / "server.pid").write_text(str(proc.pid))
    for _ in range(60):
        if _health():
            return {"running": True, "url": URL, "pid": proc.pid}
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError(f"the server did not start; see {VAR / 'server.log'}")


def stop() -> dict:
    pidf = VAR / "server.pid"
    try:
        os.killpg(int(pidf.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        pass
    pidf.unlink(missing_ok=True)
    for _ in range(20):  # uvicorn's graceful shutdown
        if not _health():
            break
        time.sleep(0.5)
    return {"running": _health()}


def _search(query: str, limit: int, path: str = "") -> list[dict]:
    start()
    q = urllib.parse.urlencode({"q": query, "limit": limit, "path": path})
    req = urllib.request.Request(f"{URL}/api/search?{q}",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:  # first call: loads the model and indexes
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise ValueError(json.load(e).get("detail", str(e))) from None


def _reindex() -> None:
    """After a write: start the server if needed and ask it to index in the
    background, so the next search does not pay for indexing."""
    try:
        start()
        req = urllib.request.Request(f"{URL}/api/sync", method="POST",
                                     headers={"Authorization": f"Bearer {TOKEN}"})
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:
        pass  # search syncs again anyway


# ---------------------------------------------------------------- tools
def _write(fn, *a, **kw) -> str:
    out = _dump(fn, *a, **kw)
    if '"error"' not in out[:20]:
        _reindex()
    return out



def _dump(fn, *a, **kw) -> str:
    try:
        return json.dumps(fn(*a, **kw), indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def create_page(path: str, content: str, format: str = "md") -> str:
    """Save a NEW page in the local qillqa notebook. Use it for notes,
    reports, call summaries and project notes you want to keep and search
    later (instead of an Artifact). `path` is hierarchical with '/', e.g.
    'project/meetings/2026-10-21'; every level is a page that can have
    subpages. `format`: 'md' (default) or 'html' (an artifact-like page with
    its own CSS/JS). Photos, videos, PDFs, .css/.js and any other files are
    added with attach_file and referenced by relative path:
    ![photo](photo.jpg), ![demo](clip.mp4) (video/audio get a player),
    [report](report.pdf), <link href="style.css">. Returns the web URL."""
    return _write(write_page, path, content, format, create=True)


@mcp.tool()
def update_page(path: str, content: str, format: str = "md") -> str:
    """Replace the content of an existing qillqa page (read it first with get_page)."""
    return _write(write_page, path, content, format, create=False)


@mcp.tool()
def get_page(path: str) -> str:
    """Read a qillqa page: title, format, content, subpages and attached files."""
    return _dump(read_page, path)


@mcp.tool()
def attach_file(path: str, source: str, name: str = "") -> str:
    """Attach a local file (photo, video, audio, PDF, .css, .js, anything) to
    the qillqa page `path` by copying it into the page folder. `source` is the
    file's path on this machine; `name` (optional) is the target file name.
    Returns the Markdown to insert into the page with update_page."""
    return _dump(attach, path, source, name)


@mcp.tool()
def list_pages(path: str = "") -> str:
    """Tree of qillqa pages under `path` ('' = everything), with titles."""
    return _dump(tree, path)


@mcp.tool()
def search(query: str, limit: int = 10, path: str = "") -> str:
    """Semantic search (by meaning, in any language) across qillqa pages.
    Every folder can act as a project: pass `path` (e.g. 'agent-trez') to
    search only that folder and everything under it; leave it empty to search
    the whole notebook. Starts the local server if needed; the first call
    takes a few seconds to load the model and index."""
    return _dump(_search, query, limit, path)


@mcp.tool()
def move_page(src: str, dst: str) -> str:
    """Move or rename a qillqa page together with all its subpages."""
    return _write(_move, src, dst)


@mcp.tool()
def delete_page(path: str, recursive: bool = False) -> str:
    """Delete a qillqa page. If it has subpages, recursive=True is required."""
    return _write(_delete, path, recursive)


# ---------------------------------------------------------------- preview
# One tool, two audiences: the PNG in `content` is what the model sees (Claude
# Code shows tool images inline), and MCP Apps hosts (Claude Desktop) also
# render PREVIEW_UI with that same image. The UI gets the image from the tool
# result, never live HTML: the host's default CSP blocks the page's own assets.

PREVIEW_UI = "ui://qillqa/preview"
CHROME = os.path.expanduser(os.environ.get("QILLQA_CHROME", "~/.local/opt/google-chrome/chrome"))
MAX_PREVIEW_HEIGHT = 8000  # ponytail: very long pages are cut here; scroll offsets if ever needed

PREVIEW_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0;font:13px/1.4 system-ui,sans-serif;color:#17201d;background:#fff}
#bar{display:flex;gap:8px;align-items:center;justify-content:space-between;padding:6px 8px}
#t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
button{font:inherit;padding:4px 10px;border:1px solid #c9d1cd;border-radius:6px;background:#f6f7f6;cursor:pointer}
img{display:block;width:100%;height:auto}</style></head><body>
<div id="bar"><span id="t">qillqa preview</span><button id="open" hidden>Open in qillqa</button></div>
<img id="img" alt="">
<script>
const [title, openBtn, shotImg] = ["t", "open", "img"].map(id => document.getElementById(id));
let seq = 0, url = "";
const pending = {};
const send = m => parent.postMessage(Object.assign({jsonrpc: "2.0"}, m), "*");
const request = (method, params) => new Promise(r => { const id = ++seq; pending[id] = r; send({id, method, params}); });
const size = () => send({method: "ui/notifications/size-changed",
  params: {width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight}});
addEventListener("message", e => {
  if (e.source !== parent) return;
  const m = e.data || {};
  if (m.id && pending[m.id]) { pending[m.id](m.result); delete pending[m.id]; return; }
  if (m.method !== "ui/notifications/tool-result") return;
  const p = m.params || {}, s = p.structuredContent || {};
  const shot = (p.content || []).find(c => c.type === "image");
  if (shot) shotImg.src = "data:" + shot.mimeType + ";base64," + shot.data;
  title.textContent = s.title || "qillqa preview";
  url = s.url || "";
  openBtn.hidden = !url;
  size();
});
shotImg.onload = size;
openBtn.onclick = () => request("ui/open-link", {url});
request("ui/initialize", {appInfo: {name: "qillqa-preview", version: "1"}, appCapabilities: {},
  protocolVersion: "2026-01-26"}).then(() => send({method: "ui/notifications/initialized"}));
</script></body></html>"""


def _signed_page_url(path: str) -> tuple[str, str, str]:
    """(absolute signed /f/ URL, page path, title). /view redirects to it."""
    d = _safe(path)
    if not _index(d):
        raise ValueError(f"no page at {path!r}")
    start()
    q = urllib.parse.urlencode({"path": _rel(d)})
    req = urllib.request.Request(f"{URL}/view?{q}", headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=10) as r:  # follows the 303 to /f/<sig>/...
        return r.geturl(), _rel(d), _title(d)


async def _screenshot(url: str, width: int) -> tuple[bytes, int, int]:
    """Full-page PNG at `width` px, plus the page's height and scroll width."""
    from playwright.async_api import async_playwright  # heavy: only when previewing

    async with async_playwright() as p:
        kw = {"executable_path": CHROME} if os.path.exists(CHROME) else {}
        browser = await p.chromium.launch(args=["--no-sandbox"], **kw)
        try:
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.goto(url, wait_until="load")
            await page.wait_for_timeout(400)  # inline chart scripts draw after load
            height, scroll = await page.evaluate(
                "[document.documentElement.scrollHeight, document.documentElement.scrollWidth]")
            png = await page.screenshot(full_page=height <= MAX_PREVIEW_HEIGHT,
                                        clip=None if height <= MAX_PREVIEW_HEIGHT else
                                        {"x": 0, "y": 0, "width": width, "height": MAX_PREVIEW_HEIGHT})
        finally:
            await browser.close()
    return png, height, scroll


@mcp.tool(meta={"ui": {"resourceUri": PREVIEW_UI}})
async def preview_page(path: str, width: int = 1100) -> CallToolResult:
    """Screenshot of a qillqa page as it renders in the browser, to check a
    page you wrote (charts, tables, layout) before sharing it. `width` in px:
    1100 desktop (default), 390 phone. Returns the PNG plus JSON with the
    page height and whether it scrolls sideways (horizontal_overflow).
    Pages taller than 8000 px are cut there. Starts the local server if needed."""
    try:
        if not 200 <= width <= 2400:
            raise ValueError("width must be between 200 and 2400 px")
        url, rel, title = await asyncio.to_thread(_signed_page_url, path)
        png, height, scroll = await _screenshot(url, width)
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))],
                              isError=True)
    info = {"path": rel, "title": title, "width": width, "height": height,
            "horizontal_overflow": scroll > width, "cut_at": MAX_PREVIEW_HEIGHT if height > MAX_PREVIEW_HEIGHT else None,
            "url": f"{URL}/#{rel}"}
    return CallToolResult(
        content=[ImageContent(type="image", data=base64.b64encode(png).decode(), mimeType="image/png"),
                 TextContent(type="text", text=json.dumps(info, ensure_ascii=False))],
        structuredContent=info)


@mcp.resource(PREVIEW_UI, name="qillqa-preview", mime_type="text/html;profile=mcp-app",
              meta={"ui": {"prefersBorder": True}})
def preview_ui() -> str:
    return PREVIEW_HTML


@mcp.tool()
def start_server() -> str:
    """Start the local qillqa web server (browse and search pages) and return its URL."""
    return _dump(start)


@mcp.tool()
def server_status() -> str:
    """Report whether the qillqa web server is up, and its URL."""
    return _dump(lambda: {"running": _health(), "url": URL, "pages_dir": str(PAGES)})


@mcp.tool()
def stop_server() -> str:
    """Stop the qillqa web server (frees all its memory)."""
    return _dump(stop)


if __name__ == "__main__":
    mcp.run()
