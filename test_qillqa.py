"""Checks without the real model: .venv/bin/python test_qillqa.py"""
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

tmp = Path(tempfile.mkdtemp())
os.environ.update(QILLQA_PAGES_DIR=str(tmp / "pages"), QILLQA_TOKEN="t0k-secret", QILLQA_MIN_SCORE="0.1")
(tmp / "pages").mkdir()

import server  # noqa: E402
import webui  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

webui.DB = tmp / "index.db"


def fake_embed(texts, query=False):
    """Deterministic bag of words: same word -> same axis."""
    out = np.zeros((len(texts), 4096), np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w{4,}", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 4096] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


webui.embed = fake_embed

# _safe: nothing escapes the notebook
for bad in ("../x", "a/../../x", "a/../.."):
    try:
        server._safe(bad)
        raise AssertionError(f"_safe accepted {bad!r}")
    except ValueError:
        pass
assert server._safe("/etc") == server.PAGES / "etc"  # absolute -> relative to the notebook
(tmp / "outside").mkdir()
(tmp / "outside" / "index.md").write_text("# outside\nsecret battery")
os.symlink(tmp / "outside", server.PAGES / "link")
try:
    server._safe("link")
    raise AssertionError("_safe followed a symlink out of the notebook")
except ValueError:
    pass
assert "link" not in webui._pages(), "the index followed a symlink out of the notebook"

# pages: create, subpage, duplicate, format switch, move, delete
server.write_page("bikes/warranty", "# Warranty\n\nThe battery has a two-year warranty.", "md", True)
server.write_page("bikes/warranty/details", "<title>Details</title><p>charge cycles</p>"
                  "<script>var battery=1</script>", "html", True)
server.write_page("kitchen", "# Recipes\n\nCeviche with lime and fresh fish.", "md", True)
try:
    server.write_page("kitchen", "x", "md", True)
    raise AssertionError("create over an existing page")
except ValueError:
    pass
t = {n["path"]: n for n in server.tree()}
assert t["bikes"]["page"] is False and t["bikes"]["children"][0]["title"] == "Warranty"
assert server.read_page("bikes/warranty/details")["title"] == "Details"
server.write_page("kitchen", "<title>Recipes</title><p>Ceviche with lime</p>", "html", False)
assert sorted(p.name for p in (server.PAGES / "kitchen").iterdir()) == ["index.html"]
try:
    server._delete("bikes")
    raise AssertionError("deleted a page with children without recursive")
except ValueError:
    pass
try:
    server._move("bikes", "bikes/inside")
    raise AssertionError("moved a page inside itself")
except ValueError:
    pass

# text and chunks
txt = webui.page_text(server.PAGES / "bikes/warranty/details/index.html")
assert "charge cycles" in txt and "var battery" not in txt, txt
cs = webui.chunks("T", "a" * 4000)
assert len(cs) == 3 and all(c.startswith("T\n\n") for c in cs)

# sync: add, unchanged, change, delete
assert webui.sync() == 3
assert webui.sync() == 0
time.sleep(0.01)
server.write_page("kitchen", "<title>Recipes</title><p>Beef stir fry</p>", "html", False)
assert webui.sync() == 1
hits = webui.search("battery warranty")
assert hits[0]["path"] == "bikes/warranty", hits
assert webui.search("zzzz qqqq xxxx") == [], "returned noise below the threshold"
assert webui._plain("## Part\n\n**Bold** `code` ![img](a.png) [link](b.md)\n```json\n{}\n```") == "Part Bold code img link {}"
server.write_page("bikesx/other", "# Other\n\nbattery warranty", "md", True)
webui.sync()
assert {h["path"] for h in webui.search("battery warranty", path="bikes")} == {"bikes/warranty"}, "scope leaked"
assert {h["path"] for h in webui.search("battery warranty", path="bikes/")} == {"bikes/warranty"}
assert "bikesx/other" in {h["path"] for h in webui.search("battery warranty")}
assert webui.search("battery warranty", path="kitchen") == []
for bad in ("nope", "../x"):
    try:
        webui.search("x", path=bad)
        raise AssertionError(f"search accepted path {bad!r}")
    except ValueError:
        pass
server._delete("bikesx", True)
server._move("bikes", "vehicles")
webui.sync()
assert {h["path"] for h in webui.search("battery warranty")} >= {"vehicles/warranty"}
assert all(not h["path"].startswith("bikes") for h in webui.search("battery"))

# web: auth
c = TestClient(webui.app)
assert c.get("/health").json()["app"] == "qillqa"
assert c.get("/api/tree").status_code == 401
assert c.get("/view?path=kitchen").status_code == 401
assert c.get("/", follow_redirects=False).status_code == 303
assert c.get("/api/tree", headers={"Authorization": "Bearer wrong"}).status_code == 401
assert c.get("/api/tree", headers={"Authorization": "Bearer t0k-secret"}).status_code == 200
assert c.post("/login", data={"token": "wrong"}, follow_redirects=False).headers["location"].startswith("/login?")
assert c.post("/login", data={"token": "tök\u00a0"}, follow_redirects=False).headers["location"].startswith("/login?")
r = c.post("/login", data={"token": " t0k-secret\n"}, follow_redirects=False)
assert r.headers["location"] == "/" and "httponly" in r.headers["set-cookie"].lower()
assert c.get("/api/tree").status_code == 200
v = c.get("/view?path=kitchen")
assert v.headers["content-security-policy"].startswith("sandbox"), v.headers
assert c.get("/view?path=../outside").status_code == 404
assert c.get("/api/search?q=stir").json()[0]["path"] == "kitchen"
assert c.get("/api/search?q=stir&path=vehicles").json() == []
assert c.get("/api/search?q=stir&path=nope").status_code == 404

# attachments and signed files
server.write_page("project x", "# Project\n\n![photo](photo.png)\n\n![clip](clip.mp4)\n\n[doc](doc.pdf)", "md", True)
src = tmp / "src"
src.mkdir()
for n, data in {"photo.png": b"\x89PNG fake", "clip.mp4": b"0123456789", "doc.pdf": b"%PDF-1.4",
                "drawing.svg": b"<svg xmlns='http://www.w3.org/2000/svg'><script>1</script></svg>"}.items():
    (src / n).write_bytes(data)
    server.attach("project x", str(src / n))
for bad in ("index.md", ".hidden", "a/b", "../x"):
    try:
        server.attach("project x", str(src / "photo.png"), bad)
        raise AssertionError(f"attach accepted the name {bad!r}")
    except ValueError:
        pass
assert server.read_page("project x")["files"] == ["clip.mp4", "doc.pdf", "drawing.svg", "photo.png"]
r = c.get("/view?path=project x", follow_redirects=False)
base = r.headers["location"]
assert r.status_code == 303 and base.startswith("/f/") and base.endswith("/project%20x/"), base
anon = TestClient(webui.app)  # no cookie: the signature is enough
page = anon.get(base)
assert page.status_code == 200 and page.headers["content-security-policy"].startswith("sandbox")
assert page.headers["referrer-policy"] == "no-referrer"
assert '<img alt="photo" src="photo.png"' in page.text and '<video controls' in page.text, page.text
md = webui._render_md("t", "## Part one\n\n[jump](#part-one) [pdf](doc.pdf)")
assert '<a href="#part-one">' in md and '<a target="_blank" href="doc.pdf">' in md, md
assert '<base' not in md
assert anon.get(base + "photo.png").headers["content-type"] == "image/png"
assert anon.get(base + "drawing.svg").headers["content-security-policy"] == "sandbox"
assert "content-security-policy" not in anon.get(base + "doc.pdf").headers
assert anon.get(base + "clip.mp4", headers={"Range": "bytes=0-3"}).status_code == 206
sig = base.split("/")[2]
assert anon.get(f"/f/{'0' * 32}/project%20x/photo.png").status_code == 404, "forged signature"
assert anon.get(f"/f/{sig}/kitchen/").status_code == 404, "signature of another page"
assert anon.get(f"/f/{sig}/project%20x/%2e%2e/kitchen/").status_code == 404, ".. inside /f"
server.write_page("pa/b", "# b", "md", True)
server.write_page("pa/bc", "# bc", "md", True)
assert anon.get(f"/f/{webui._sig('pa/b')}/pa/bc/").status_code == 404, "string prefix pa/b -> pa/bc"
assert anon.get(f"/f/{webui._sig('')}/kitchen/").status_code == 404, "root signature"
assert anon.get("/api/tree").status_code == 401

# checklists: ☐/☑/◐ in a md page are clickable and table rows can be dragged
CL = """# Plan ☐ legend: ☑ done

| ✓ | Task |
|---|---|
| ☐ | write tests |
| ◐ | ship it ☐ |

```
| ☐ | inside a fence |
|---|---|
| ☐ | not a row |
```

| ✓ | Done |
|---|---|
| ☑ | read the code |

- ☐ loose box
"""
e = webui.edit_checklist
assert e(CL, "check", 0).count("☑ | write tests") == 1
assert "| ☑ | ship it ☐ |" in e(CL, "check", 1), "◐ -> ☑"
assert "| ◐ | ship it ☑ |" in e(CL, "check", 2), "second box on the same line"
assert "| ☐ | read the code" in e(CL, "check", 3), "☑ -> ☐"
assert "- ☑ loose box" in e(CL, "check", 4), "list item box"
moved = e(CL, "move", 0, to=3, after=True)  # 'write tests' -> under 'read the code' in the other table
assert moved.index("read the code") < moved.index("write tests") and moved.count("write tests") == 1, moved
assert moved.split("| ✓ | Task |")[1].split("```")[0].count("|") == 6, "source table lost the row"
assert e(CL, "move", 1, to=0).index("ship it") < e(CL, "move", 1, to=0).index("write tests"), "move before"
for bad in (dict(op="move", i=4, to=0), dict(op="move", i=0, to=4), dict(op="move", i=0, to=5), dict(op="check", i=99), dict(op="x", i=0)):
    try:
        e(CL, bad.pop("op"), **bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
assert webui._body_rows(CL.split("\n")) == {4, 5, 15}, webui._body_rows(CL.split("\n"))

html_ = webui._render_md("t", CL + "\n<script>alert(1)</script>", nonce="N0nce")
assert [int(x) for x in re.findall(r'data-i="(\d+)"', html_)] == list(range(5)), "boxes numbered in source order"
assert html_.count('class="grip"') == 3, "one grip per table row, none in fences or lists"
assert "<h1>Plan ☐ legend: ☑ done</h1>" in html_.replace(' id="plan-legend-done"', ""), "a box in prose stays text"
assert '<script nonce="N0nce">' in html_ and 'data-v="' in html_
assert "\ue000" not in html_ and '<div class="tw"><table>' in html_
assert 'class="cb"' not in webui._render_md("t", CL), "without a nonce the page stays static"

server.write_page("todo", CL, "md", True)
live = anon.get(c.get("/view?path=todo", follow_redirects=False).headers["location"])
csp = live.headers["content-security-policy"]
assert "allow-scripts" in csp and "script-src 'nonce-" in csp, csp
nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
assert f'<script nonce="{nonce}">' in live.text
assert "allow-scripts" not in c.get("/view?path=pa/b").headers["content-security-policy"], "plain md stays script-free"
v0 = re.search(r'data-v="(\w+)"', live.text).group(1)
body = {"path": "todo", "op": "check", "i": 0, "v": v0}
assert anon.post("/api/page/edit", json=body).status_code == 401
assert c.post("/api/page/edit", data="x", headers={"content-type": "text/plain"}).status_code == 415
r = c.post("/api/page/edit", json=body)
assert r.status_code == 200 and "| ☑ | write tests |" in server.read_page("todo")["content"], r.text
assert c.post("/api/page/edit", json=body).status_code == 409, "stale version must not overwrite"
r = c.post("/api/page/edit", json={**body, "op": "move", "i": 0, "to": 3, "after": True, "v": r.json()["v"]})
assert r.status_code == 200, r.text
assert c.post("/api/page/edit", json={**body, "path": "bikes/warranty/details", "v": "x"}).status_code == 400, "html page"
assert c.post("/api/page/edit", json={**body, "path": "../outside", "v": "x"}).status_code == 400

print("OK: all checks pass")
