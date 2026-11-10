"""Share links (/s/<token>): .venv/bin/python test_share.py"""
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ.update(QILLQA_PAGES_DIR=str(tmp / "pages"), QILLQA_VAR_DIR=str(tmp / "var"),
                  QILLQA_TOKEN="t0k-secret", QILLQA_PUBLIC_URL="")
(tmp / "pages").mkdir()

import server  # noqa: E402
import webui  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

server.start = lambda: {"running": True}  # the MCP tool must not launch a real server here

server.write_page("fin", "# Finanzas\n\nIngresos ![g](chart.png)\n\n| ok |\n|---|\n| ☐ tarea |", "md", True)
(server.PAGES / "fin" / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
(server.PAGES / "fin" / "assets").mkdir()
(server.PAGES / "fin" / "assets" / "data.json").write_text("{}")
server.write_page("fin/sub", "<title>Sub</title><p>sub secret</p><script>var x=1</script>", "html", True)
(server.PAGES / "fin" / "sub" / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n")
server.write_page("other", "# Other\n\nnot shared", "md", True)
(server.PAGES / "fin" / ".secret").write_text("hidden")
server.write_page("priv", "# Privado\n\nprivate payroll", "md", True)
server.write_page("fin/sub/deep", "# Deep\n\nlinks: [up](#fin) [other](#other) [sub](#fin/sub)", "md", True)

owner = TestClient(webui.app, cookies={webui.COOKIE: webui.SESSION})
anon = TestClient(webui.app)

# --- creating: only the owner, only pages, bounded duration
assert anon.post("/api/share", json={"path": "fin"}).status_code == 401
assert anon.get("/api/shares").status_code == 401
assert anon.post("/api/share/revoke", json={"id": "x"}).status_code == 401
assert owner.post("/api/share", json={"path": "", "hours": 1}).status_code == 404, "root is not a page"
assert owner.post("/api/share", json={"path": "../etc", "hours": 1}).status_code == 404
assert owner.post("/api/share", json={"path": "fin", "hours": 0}).status_code == 404
assert owner.post("/api/share", json={"path": "fin", "hours": 24 * 91}).status_code == 404
for bad in ({"path": "fin", "subpages": "false"}, {"path": "fin", "hours": "1"}, {"path": "fin", "hours": True},
            {"path": "fin", "label": 3}, {"path": ["fin"]}):
    assert owner.post("/api/share", json=bad).status_code == 400, bad
r = owner.post("/api/share", json={"path": "fin", "hours": 1, "label": "  Julio\n <b>x</b> "},
               headers={"host": "jordi-omen.tail.ts.net"})
assert r.status_code == 200, r.text
sh = r.json()
assert sh["url"].startswith("https://jordi-omen.tail.ts.net/s/"), "ts.net link must be https"
assert sh["label"] == "Julio <b>x</b>" and not sh["subpages"] and sh["state"] == "active"
tok = sh["token"]
assert re.fullmatch(r"[0-9a-f]{16}\.[0-9a-f]{32}", tok)
row = sqlite3.connect(server.SHARES_DB).execute("select * from shares").fetchone()
assert tok.split(".")[1] not in " ".join(map(str, row)), "the secret part must not be stored"

# --- the viewer
v = anon.get(f"/s/{tok}")
assert v.status_code == 200 and "Finanzas" in v.text
assert "/f/" not in v.text, "the viewer must never hand out a permanent /f/ signature"
assert "frame-ancestors 'none'" in v.headers["content-security-policy"]
assert v.headers["cache-control"] == "no-store" and "noindex" in v.headers["x-robots-tag"]
assert "<select" not in v.text, "no subpage list without subpages"
assert "Julio" not in v.text, "the label is for the owner, not shown to the visitor"

base = f"/s/{tok}/p/fin/"
p = anon.get(base)
assert p.status_code == 200 and "Ingresos" in p.text
assert p.headers["content-security-policy"].startswith("sandbox"), "the page runs sandboxed"
assert "data-cap" not in p.text and "button class=\"cb\"" not in p.text, "read-only: no checklist capability"
assert anon.get(base + "chart.png").status_code == 200
assert anon.get(base + "chart.png").headers["cache-control"] == "no-store"
assert anon.get(base + "assets/data.json").status_code == 200, "asset folders (no page) are part of the page"
assert anon.get(base + "sub/").status_code == 404, "subpage not shared"
assert anon.get(base + "sub/x.png").status_code == 404, "file of a subpage not shared"
assert anon.get(f"/s/{tok}/p/other/").status_code == 404, "another page"
assert anon.get(f"/s/{tok}/p/fin/%2e%2e/other/").status_code == 404, ".. inside /s"
assert anon.get(f"/s/{tok}/p/").status_code == 404
assert anon.get(f"/s/{tok}/p/fin/%2e%2e/fin/chart.png").status_code == 404, ".. back into the scope"
assert anon.get(f"/s/{tok}/p/fin/.secret").status_code == 404, "hidden file"
assert anon.get(f"/s/{tok}/p/fin", follow_redirects=False).status_code == 307, "relative links need the trailing slash"

# --- forged / unknown tokens
sid, mac = tok.split(".")
for bad in (f"{sid}.{'0' * 32}", f"{'0' * 16}.{server._share_mac('0' * 16)}", sid, "x", f"{sid}.{mac}x"):
    assert anon.get(f"/s/{bad}").status_code == 410, bad
    assert anon.get(f"/s/{bad}/p/fin/").status_code == 410, bad

# --- counting: the viewer counts, its files do not
uses = {s["id"]: s["uses"] for s in owner.get("/api/shares?path=fin").json()}
assert uses[sid] == 1, uses

# --- expiry is checked on every request, files included
with sqlite3.connect(server.SHARES_DB) as db:
    db.execute("update shares set expires=? where id=?", (time.time() - 1, sid))
assert anon.get(f"/s/{tok}").status_code == 410
assert anon.get(f"/s/{tok}/p/fin", follow_redirects=False).status_code == 410, "no redirect for a dead link"
assert anon.get(base).status_code == 410 and anon.get(base + "chart.png").status_code == 410
assert [s["state"] for s in owner.get("/api/shares?path=fin").json()] == ["expired"]

# --- subpages + revocation
s2 = server.share_create("fin", 2, True, "equipo")
t2 = s2["token"]
v2 = anon.get(f"/s/{t2}")
assert "<select" in v2.text and "Sub" in v2.text
assert anon.get(f"/s/{t2}/p/fin/sub/").status_code == 200
assert anon.get(f"/s/{t2}/p/fin/sub/x.png").status_code == 200
assert anon.get(f"/s/{t2}/p/other/").status_code == 404
deep = anon.get(f"/s/{t2}/p/fin/sub/deep/").text
assert f'href="/s/{t2}/p/fin/"' in deep and f'href="/s/{t2}/p/fin/sub/"' in deep, "links to shared pages stay in the share"
assert 'href="/#other"' not in deep and "/p/other/" not in deep, "never a way out to the notebook"
assert deep.count('class="off"') == 1, "a link to a page outside the share is inert"
assert owner.post("/api/share/revoke", json={"id": s2["id"]}).json()["state"] == "revoked"
assert anon.get(f"/s/{t2}/p/fin/sub/").status_code == 410, "revoked at once"
assert owner.post("/api/share/revoke", json={"id": "nope"}).status_code == 404

# --- the page moved away: the link stops (it is bound to the path)
s3 = server.share_create("other", 1)
server._move("other", "elsewhere")
assert anon.get(f"/s/{s3['token']}").status_code == 410

# --- rotating the token kills every link
s4 = server.share_create("fin", 1)
real = server.TOKEN
server.TOKEN = "rotated"
try:
    assert anon.get(f"/s/{s4['token']}").status_code == 410
finally:
    server.TOKEN = real
assert anon.get(f"/s/{s4['token']}").status_code == 200

# --- symlinks: an index or a folder that points elsewhere opens nothing
os.symlink(server.PAGES / "priv" / "index.md", server.PAGES / "fin" / "sub" / "sneak.md")
(server.PAGES / "fin" / "alias").symlink_to(server.PAGES / "priv")
(server.PAGES / "fin" / "fake").mkdir()
(server.PAGES / "fin" / "fake" / "index.md").symlink_to(server.PAGES / "priv" / "index.md")
s5 = server.share_create("fin", 1, True)
v5 = anon.get(f"/s/{s5['token']}").text
assert "Privado" not in v5 and "alias" not in v5 and "fake" not in v5, "no titles through symlinks"
for path in ("fin/alias/", "fin/fake/", "fin/sub/sneak.md"):
    r = anon.get(f"/s/{s5['token']}/p/{path}")
    assert r.status_code == 404 and "payroll" not in r.text, path
server.write_page("linked", "x", "md", True)
(server.PAGES / "linked" / "index.md").unlink()
(server.PAGES / "linked" / "index.md").symlink_to(server.PAGES / "priv" / "index.md")
assert owner.post("/api/share", json={"path": "linked"}).status_code == 404, "symlinked index can't be shared"

# --- a page deleted or moved and created again at the same path: the old link stays dead
s6 = server.share_create("priv", 1)
server._delete("priv", True)
server.write_page("priv", "# Nuevo\n\nnew page, same path", "md", True)
assert anon.get(f"/s/{s6['token']}").status_code == 410 and anon.get(f"/s/{s6['token']}/p/priv/").status_code == 410
assert [s["state"] for s in server.share_list("priv") if s["id"] == s6["id"]] == ["revoked"]
s7 = server.share_create("priv", 1)
server._move("priv", "priv2")
server.write_page("priv", "# Otra\n\nanother page", "md", True)
assert anon.get(f"/s/{s7['token']}").status_code == 410

# --- changed by hand, outside the MCP: the folder identity still stops the old link
import shutil  # noqa: E402
server.write_page("hand", "# Hand\n\nv1", "md", True)
s8 = server.share_create("hand", 1)
shutil.move(str(server.PAGES / "hand"), str(server.PAGES / "hand-old"))  # renamed by hand, not via _move
server.write_page("hand", "# Hand v2\n\nanother page at the same path", "md", True)
assert anon.get(f"/s/{s8['token']}").status_code == 410
server.write_page("swap", "# Swap\n\nv1", "md", True)
s9 = server.share_create("swap", 1)
(server.PAGES / "swap" / "index.md").unlink()
(server.PAGES / "swap" / "index.md").symlink_to(server.PAGES / "priv" / "index.md")
assert anon.get(f"/s/{s9['token']}").status_code == 410, "index swapped for a symlink after sharing"
assert anon.get(f"/s/{s9['token']}/p/swap/").status_code == 404

# --- MCP tools
import json  # noqa: E402
out = json.loads(server.share_page("fin", 1, False, "mcp"))
assert out["url"].startswith(server.PUBLIC_URL + "/s/") and out["label"] == "mcp"
assert any(s["id"] == out["id"] for s in json.loads(server.list_shares("fin")))
assert json.loads(server.revoke_share(out["id"]))["state"] == "revoked"
assert "error" in json.loads(server.share_page("missing", 1))

# --- the permanent /f/ signature still works for the owner (unchanged)
loc = owner.get("/view?path=fin", follow_redirects=False).headers["location"]
assert loc.startswith("/f/") and "data-cap" in anon.get(loc).text, "owner view keeps live checklists"

print("OK: share checks pass")
