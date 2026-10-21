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
