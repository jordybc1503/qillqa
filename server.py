#!/usr/bin/env python3
"""qillqa MCP (stdio) — cuaderno local de páginas anidadas.

Una página es una carpeta bajo QILLQA_PAGES_DIR con index.md o index.html;
sus subcarpetas son subpáginas. Las escrituras van directo al disco (sin
servidor) y después se le pide al servidor web local (webui.py) que indexe en
segundo plano; el MCP lo arranca bajo demanda (~0,5 s).

No importar torch/numpy/sentence_transformers aquí: este módulo carga en el
handshake del MCP y webui.py lo importa.
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


# ---------------------------------------------------------------- páginas

def _safe(path: str) -> Path:
    """Ruta de página → carpeta dentro de PAGES. Rechaza todo lo que salga."""
    p = (PAGES / (path or "").strip().strip("/")).resolve()
    if p != PAGES and PAGES not in p.parents:
        raise ValueError(f"ruta fuera del cuaderno: {path!r}")
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
        raise ValueError(f"no existe: {path!r}")
    return [node(c) for c in _children(root)]


def read_page(path: str) -> dict:
    d = _safe(path)
    idx = _index(d)
    if not idx:
        raise ValueError(f"no hay página en {path!r}")
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
    """Copia un archivo local a la carpeta de la página."""
    d = _safe(path)
    if d == PAGES or not d.is_dir():
        raise ValueError(f"no existe la página {path!r}; créala primero")
    src = Path(os.path.expanduser(source))
    if not src.is_file():
        raise ValueError(f"no existe el archivo {source!r}")
    name = name or src.name
    if "/" in name or "\\" in name or name.startswith(".") or name in INDEX_NAMES:
        raise ValueError(f"nombre de archivo no válido: {name!r}")
    shutil.copyfile(src, d / name)
    ref = urllib.parse.quote(name)
    md = f"![{name}]({ref})" if Path(name).suffix.lower() in _MEDIA else f"[{name}]({ref})"
    return {"page": _rel(d), "file": name, "markdown": md}


def write_page(path: str, content: str, fmt: str, create: bool) -> dict:
    if fmt not in ("md", "html"):
        raise ValueError("format debe ser 'md' o 'html'")
    d = _safe(path)
    if d == PAGES:
        raise ValueError("indica una ruta de página, p.ej. 'proyecto/apunte'")
    exists = _index(d) is not None
    if create and exists:
        raise ValueError(f"ya existe {path!r}; usa update_page")
    if not create and not exists:
        raise ValueError(f"no existe {path!r}; usa create_page")
    d.mkdir(parents=True, exist_ok=True)
    for n in INDEX_NAMES:  # cambiar de formato no deja el index viejo
        if n != f"index.{fmt}":
            (d / n).unlink(missing_ok=True)
    (d / f"index.{fmt}").write_text(content)
    return {"path": _rel(d), "title": _title(d), "url": f"{URL}/#{_rel(d)}"}


def _move(src: str, dst: str) -> dict:
    s, t = _safe(src), _safe(dst)
    if s == PAGES or not s.is_dir():
        raise ValueError(f"no existe {src!r}")
    if t.exists():
        raise ValueError(f"ya existe {dst!r}")
    if t == s or s in t.parents:
        raise ValueError("no se puede mover una página dentro de sí misma")
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(t))
    return {"path": _rel(t), "url": f"{URL}/#{_rel(t)}"}


def _delete(path: str, recursive: bool = False) -> dict:
    d = _safe(path)
    if d == PAGES or not d.is_dir():
        raise ValueError(f"no existe {path!r}")
    kids = _children(d)
    if kids and not recursive:
        raise ValueError(f"{path!r} tiene {len(kids)} subpáginas; pasa recursive=True")
    shutil.rmtree(d)
    return {"deleted": _rel(d)}


# ---------------------------------------------------------------- servidor

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
        raise RuntimeError("QILLQA_TOKEN vacío en .env; corre setup.sh o pon un token")
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
    raise RuntimeError(f"el servidor no arrancó; mira {VAR / 'server.log'}")


def stop() -> dict:
    pidf = VAR / "server.pid"
    try:
        os.killpg(int(pidf.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        pass
    pidf.unlink(missing_ok=True)
    for _ in range(20):  # apagado ordenado de uvicorn
        if not _health():
            break
        time.sleep(0.5)
    return {"running": _health()}


def _search(query: str, limit: int) -> list[dict]:
    start()
    q = urllib.parse.urlencode({"q": query, "limit": limit})
    req = urllib.request.Request(f"{URL}/api/search?{q}",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=900) as r:  # 1ª vez: carga modelo + indexa
        return json.load(r)


def _reindex() -> None:
    """Tras escribir: enciende el servidor si hace falta y le pide indexar en
    segundo plano, para que la próxima búsqueda no pague la indexación."""
    try:
        start()
        req = urllib.request.Request(f"{URL}/api/sync", method="POST",
                                     headers={"Authorization": f"Bearer {TOKEN}"})
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:
        pass  # la búsqueda vuelve a sincronizar igual


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
    """Guarda una página NUEVA en el cuaderno local qillqa. Úsalo para apuntes,
    informes, resúmenes de llamadas y notas de proyectos que quieras conservar
    y buscar después (en lugar de un Artifact). `path` es jerárquico con '/',
    p.ej. 'trez/edwin/llamada-2026-10-21'; cada nivel es una página que puede
    tener subpáginas. `format`: 'md' (por defecto) o 'html' (página tipo
    artifact con su CSS/JS). Fotos, vídeos, PDFs, .css/.js y demás archivos
    se añaden con attach_file y se referencian con ruta relativa:
    ![foto](foto.jpg), ![demo](clip.mp4) (vídeo/audio se reproducen),
    [informe](informe.pdf), <link href="estilo.css">. Devuelve la URL web."""
    return _write(write_page, path, content, format, create=True)


@mcp.tool()
def update_page(path: str, content: str, format: str = "md") -> str:
    """Reemplaza el contenido de una página existente de qillqa (lee antes con get_page)."""
    return _write(write_page, path, content, format, create=False)


@mcp.tool()
def get_page(path: str) -> str:
    """Lee una página de qillqa: título, formato, contenido y subpáginas."""
    return _dump(read_page, path)


@mcp.tool()
def attach_file(path: str, source: str, name: str = "") -> str:
    """Adjunta un archivo local (foto, vídeo, audio, PDF, .css, .js, lo que sea)
    a la página `path` de qillqa, copiándolo a su carpeta. `source` es la ruta
    del archivo en este equipo; `name` (opcional) es el nombre destino.
    Devuelve el Markdown para insertarlo en la página con update_page."""
    return _dump(attach, path, source, name)


@mcp.tool()
def list_pages(path: str = "") -> str:
    """Árbol de páginas de qillqa bajo `path` ('' = todo), con títulos."""
    return _dump(tree, path)


@mcp.tool()
def search(query: str, limit: int = 10) -> str:
    """Búsqueda semántica (por significado, en cualquier idioma) en todas las
    páginas de qillqa. Arranca el servidor local si hace falta; la primera vez
    tarda en cargar el modelo e indexar."""
    return _dump(_search, query, limit)


@mcp.tool()
def move_page(src: str, dst: str) -> str:
    """Mueve o renombra una página de qillqa con todas sus subpáginas."""
    return _write(_move, src, dst)


@mcp.tool()
def delete_page(path: str, recursive: bool = False) -> str:
    """Borra una página de qillqa. Si tiene subpáginas exige recursive=True."""
    return _write(_delete, path, recursive)


@mcp.tool()
def start_server() -> str:
    """Arranca el servidor web local de qillqa (navegar y buscar páginas) y devuelve su URL."""
    return _dump(start)


@mcp.tool()
def server_status() -> str:
    """Dice si el servidor web de qillqa está arriba y su URL."""
    return _dump(lambda: {"running": _health(), "url": URL, "pages_dir": str(PAGES)})


@mcp.tool()
def stop_server() -> str:
    """Detiene el servidor web de qillqa (libera la RAM del modelo)."""
    return _dump(stop)


if __name__ == "__main__":
    mcp.run()
