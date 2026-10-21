"""Checks sin modelo real: .venv/bin/python test_qillqa.py"""
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

tmp = Path(tempfile.mkdtemp())
os.environ.update(QILLQA_PAGES_DIR=str(tmp / "pages"), QILLQA_TOKEN="t0k-secreto", QILLQA_MIN_SCORE="0.1")
(tmp / "pages").mkdir()

import server  # noqa: E402
import webui  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

webui.DB = tmp / "index.db"


def fake_embed(texts, query=False):
    """Bolsa de palabras determinista: misma palabra ⇒ mismo eje."""
    out = np.zeros((len(texts), 4096), np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w{4,}", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 4096] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


webui.embed = fake_embed

# _safe: nada sale del cuaderno
for bad in ("../x", "a/../../x", "a/../.."):
    try:
        server._safe(bad)
        raise AssertionError(f"_safe aceptó {bad!r}")
    except ValueError:
        pass
assert server._safe("/etc") == server.PAGES / "etc"  # absoluta ⇒ relativa al cuaderno
(tmp / "outside").mkdir()
(tmp / "outside" / "index.md").write_text("# fuera\nsecreto bateria")
os.symlink(tmp / "outside", server.PAGES / "link")
try:
    server._safe("link")
    raise AssertionError("_safe siguió un symlink hacia fuera")
except ValueError:
    pass
assert "link" not in webui._pages(), "el índice siguió un symlink hacia fuera"

# páginas: crear, subpágina, duplicado, cambio de formato, mover, borrar
server.write_page("motos/garantia", "# Garantía\n\nLa bateria dura dos años de garantia.", "md", True)
server.write_page("motos/garantia/detalle", "<title>Detalle</title><p>ciclos de carga</p>"
                  "<script>var bateria=1</script>", "html", True)
server.write_page("cocina", "# Recetas\n\nCeviche con limon y pescado fresco.", "md", True)
try:
    server.write_page("cocina", "x", "md", True)
    raise AssertionError("create sobre una página existente")
except ValueError:
    pass
t = {n["path"]: n for n in server.tree()}
assert t["motos"]["page"] is False and t["motos"]["children"][0]["title"] == "Garantía"
assert server.read_page("motos/garantia/detalle")["title"] == "Detalle"
server.write_page("cocina", "<title>Recetas</title><p>Ceviche con limon</p>", "html", False)
assert sorted(p.name for p in (server.PAGES / "cocina").iterdir()) == ["index.html"]
try:
    server._delete("motos")
    raise AssertionError("borró una página con hijos sin recursive")
except ValueError:
    pass
try:
    server._move("motos", "motos/dentro")
    raise AssertionError("movió una página dentro de sí misma")
except ValueError:
    pass

# texto y chunks
txt = webui.page_text(server.PAGES / "motos/garantia/detalle/index.html")
assert "ciclos de carga" in txt and "var bateria" not in txt, txt
cs = webui.chunks("T", "a" * 4000)
assert len(cs) == 3 and all(c.startswith("T\n\n") for c in cs)

# sync: alta, sin cambios, cambio, baja
assert webui.sync() == 3
assert webui.sync() == 0
time.sleep(0.01)
server.write_page("cocina", "<title>Recetas</title><p>Lomo saltado</p>", "html", False)
assert webui.sync() == 1
hits = webui.search("bateria garantia")
assert hits[0]["path"] == "motos/garantia", hits
assert webui.search("zzzz qqqq xxxx") == [], "devolvió ruido bajo el umbral"
server._move("motos", "vehiculos")
webui.sync()
assert {h["path"] for h in webui.search("bateria garantia")} >= {"vehiculos/garantia"}
assert all(not h["path"].startswith("motos") for h in webui.search("bateria"))

# web: auth
c = TestClient(webui.app)
assert c.get("/health").json()["app"] == "qillqa"
assert c.get("/api/tree").status_code == 401
assert c.get("/view?path=cocina").status_code == 401
assert c.get("/", follow_redirects=False).status_code == 303
assert c.get("/api/tree", headers={"Authorization": "Bearer mal"}).status_code == 401
assert c.get("/api/tree", headers={"Authorization": "Bearer t0k-secreto"}).status_code == 200
assert c.post("/login", data={"token": "mal"}, follow_redirects=False).headers["location"].startswith("/login?")
assert c.post("/login", data={"token": "tök\u00a0"}, follow_redirects=False).headers["location"].startswith("/login?")
r = c.post("/login", data={"token": " t0k-secreto\n"}, follow_redirects=False)
assert r.headers["location"] == "/" and "httponly" in r.headers["set-cookie"].lower()
assert c.get("/api/tree").status_code == 200
v = c.get("/view?path=cocina")
assert v.headers["content-security-policy"].startswith("sandbox"), v.headers
assert c.get("/view?path=../outside").status_code == 404
assert c.get("/api/search?q=lomo").json()[0]["path"] == "cocina"

# adjuntos y archivos firmados
server.write_page("proyecto x", "# Proyecto\n\n![foto](foto.png)\n\n![clip](clip.mp4)\n\n[doc](doc.pdf)", "md", True)
src = tmp / "src"
src.mkdir()
for n, data in {"foto.png": b"\x89PNG fake", "clip.mp4": b"0123456789", "doc.pdf": b"%PDF-1.4",
                "dibujo.svg": b"<svg xmlns='http://www.w3.org/2000/svg'><script>1</script></svg>"}.items():
    (src / n).write_bytes(data)
    server.attach("proyecto x", str(src / n))
for bad in ("index.md", ".oculto", "a/b", "../x"):
    try:
        server.attach("proyecto x", str(src / "foto.png"), bad)
        raise AssertionError(f"attach aceptó el nombre {bad!r}")
    except ValueError:
        pass
assert server.read_page("proyecto x")["files"] == ["clip.mp4", "dibujo.svg", "doc.pdf", "foto.png"]
r = c.get("/view?path=proyecto x", follow_redirects=False)
base = r.headers["location"]
assert r.status_code == 303 and base.startswith("/f/") and base.endswith("/proyecto%20x/"), base
anon = TestClient(webui.app)  # sin cookie: la firma basta
page = anon.get(base)
assert page.status_code == 200 and page.headers["content-security-policy"].startswith("sandbox")
assert page.headers["referrer-policy"] == "no-referrer"
assert '<img alt="foto" src="foto.png"' in page.text and '<video controls' in page.text, page.text
assert anon.get(base + "foto.png").headers["content-type"] == "image/png"
assert anon.get(base + "dibujo.svg").headers["content-security-policy"] == "sandbox"
assert "content-security-policy" not in anon.get(base + "doc.pdf").headers
assert anon.get(base + "clip.mp4", headers={"Range": "bytes=0-3"}).status_code == 206
sig = base.split("/")[2]
assert anon.get(f"/f/{'0' * 32}/proyecto%20x/foto.png").status_code == 404, "firma falsa"
assert anon.get(f"/f/{sig}/cocina/").status_code == 404, "firma de otra página"
assert anon.get(f"/f/{sig}/proyecto%20x/%2e%2e/cocina/").status_code == 404, ".. dentro de /f"
server.write_page("pa/b", "# b", "md", True)
server.write_page("pa/bc", "# bc", "md", True)
assert anon.get(f"/f/{webui._sig('pa/b')}/pa/bc/").status_code == 404, "prefijo de texto pa/b → pa/bc"
assert anon.get(f"/f/{webui._sig('')}/cocina/").status_code == 404, "firma de la raíz"
assert anon.get("/api/tree").status_code == 401

print("OK: todos los checks pasan")
