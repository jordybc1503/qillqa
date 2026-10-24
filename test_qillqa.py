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

# checklists: ☐/☑/◐ in a table row or at the start of a list item are clickable, rows drag
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
# box numbers count every glyph in the source: 0,1 legend · 2 tests · 3,4 ship · 5,6 fence · 7 read · 8 loose
assert webui._kinds(CL) == {2: "row", 3: "row", 4: "row", 7: "row", 8: "item"}, webui._kinds(CL)
e = webui.edit_checklist
assert "| ☑ | write tests |" in e(CL, "check", 2)
assert "| ☑ | ship it ☐ |" in e(CL, "check", 3), "◐ -> ☑"
assert "| ◐ | ship it ☑ |" in e(CL, "check", 4), "second box on the same line"
assert "| ☐ | read the code" in e(CL, "check", 7) and "- ☑ loose box" in e(CL, "check", 8)
moved = e(CL, "move", 2, to=7, after=True)  # 'write tests' -> under 'read the code' in the other table
assert moved.index("read the code") < moved.index("write tests") and moved.count("write tests") == 1, moved
assert moved.split("| ✓ | Task |")[1].split("```")[0].count("write") == 0, "source table kept the row"
assert e(CL, "move", 3, to=2).index("ship it") < e(CL, "move", 3, to=2).index("write tests"), "move before"
for bad in (dict(op="check", i=0), dict(op="check", i=5), dict(op="move", i=8, to=2), dict(op="move", i=2, to=8),
            dict(op="move", i=2, to=99), dict(op="check", i=99), dict(op="x", i=2)):
    try:
        e(CL, bad.pop("op"), **bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
crlf = CL.replace("\n", "\r\n")
assert e(crlf, "check", 2).count("\r\n") == crlf.count("\r\n"), "CRLF kept on check"
m2 = e(crlf, "move", 2, to=7, after=True)
assert m2.count("\r\n") == crlf.count("\r\n") and "\n" not in m2.replace("\r\n", ""), "CRLF kept on move"
assert e("| a | b |\n|---|---|\n| ☐ | x |", "move", 0, to=0) == "| a | b |\n|---|---|\n| ☐ | x |"

# what markdown itself decides, not a regex: tables without outer pipes / compact separators are
# tables; an indented table is code; a box in a link url or image alt is no button
odd = {"a | b\n---|---\n☐ | task": {0: "row"}, "|a|b|\n|-|-|\n|☐|task|": {0: "row"},
       "    | a | b |\n    |---|---|\n    | ☐ | x |": {},
       "````\n```\n- ☐ code\n```\n````\n\n- ☐ real": {1: "item"},
       "- ☐ [link](https://example.com/☐) ☐": {0: "item"}, "- ![☐](pic.png) task": {},
       "text ☐ in a sentence": {}, "- a ☐ late box": {}, "1. ☐ ordered": {0: "item"}}
for src, want in odd.items():
    assert webui._kinds(src) == want, (src, webui._kinds(src))
link = webui._render_md("t", "- ☐ [link](https://example.com/☐) ☐", cap="c", nonce="n")
assert 'href="https://example.com/☐"' in link and link.count('class="cb"') == 1, link
alt = webui._render_md("t", "- ☐ ok\n- ![☐](pic.png)", cap="c", nonce="n")
assert 'alt="☐"' in alt and alt.count('class="cb"') == 1, alt
pua = webui._render_md("t", "0☐\n\n- ☐ actual", cap="c", nonce="n")
assert pua.count('data-i="') == 1 and 'data-i="1"' in pua, "a literal marker in the source is not a button"

html_ = webui._render_md("t", CL + "\n<script>alert(1)</script>", cap="C4p", nonce="N0nce")
assert [int(x) for x in re.findall(r'data-i="(\d+)"', html_)] == [2, 3, 4, 7, 8]
assert html_.count('class="grip"') == 3, "one grip per table row, none in fences or lists"
assert all("<tr>" in r for r in html_.split('class="grip"')[:-1]), "every grip sits in a row"
assert '<script nonce="N0nce">' in html_ and 'data-cap="C4p"' in html_ and "" not in html_
assert "<h1" in html_ and "Plan ☐ legend: ☑ done</h1>" in html_, "a box in prose stays text"
assert '<div class="tw"><table>' in html_
assert 'class="cb"' not in webui._render_md("t", CL), "without a capability the page stays static"

server.write_page("todo", CL, "md", True)
live = anon.get(c.get("/view?path=todo", follow_redirects=False).headers["location"])
csp = live.headers["content-security-policy"]
assert "allow-scripts" in csp and "script-src 'nonce-" in csp, csp
nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
assert f'<script nonce="{nonce}">' in live.text
assert "allow-scripts" not in c.get("/view?path=pa/b").headers["content-security-policy"], "plain md stays script-free"
cap = re.search(r'data-cap="([\w-]+)"', live.text).group(1)
body = {"path": "todo", "op": "check", "i": 2, "cap": cap}
assert anon.post("/api/page/edit", json=body).status_code == 401
assert c.post("/api/page/edit", data="x", headers={"content-type": "text/plain"}).status_code == 415
assert c.post("/api/page/edit", json={**body, "cap": "guessed"}).status_code == 403, "no capability"
assert c.post("/api/page/edit", json={**body, "path": "kitchen"}).status_code in (400, 403), "cap of another page"
server.write_page("todo2", CL, "md", True)
assert c.post("/api/page/edit", json={**body, "path": "todo2"}).status_code == 403, "cap bound to its page"
r = c.post("/api/page/edit", json=body)
assert r.status_code == 200 and "| ☑ | write tests |" in server.read_page("todo")["content"], r.text
assert c.post("/api/page/edit", json=body).status_code == 409, "stale render must not overwrite"
cap = re.search(r'data-cap="([\w-]+)"', anon.get(c.get("/view?path=todo", follow_redirects=False)
                                               .headers["location"]).text).group(1)
r = c.post("/api/page/edit", json={**body, "op": "move", "i": 2, "to": 7, "after": True, "cap": cap})
assert r.status_code == 200, r.text
assert c.post("/api/page/edit", json={**body, "path": "bikes/warranty/details"}).status_code == 400, "html page"
assert c.post("/api/page/edit", json={**body, "path": "../outside"}).status_code == 400

# round 2: raw html table cells click but do not move; different columns refused; lone \r is a line
raw = "<table><tbody><tr>\n<td>☐ one</td>\n<td>d1</td>\n</tr><tr>\n<td>☐ two</td>\n<td>d2</td>\n</tr></tbody></table>"
assert webui._kinds(raw) == {0: "cell", 1: "cell"} and "☑ one" in e(raw, "check", 0)
for bad in ((raw, 0, 1),):
    try:
        e(bad[0], "move", bad[1], to=bad[2])
        raise AssertionError("moved a raw html cell")
    except ValueError:
        pass
assert 'class="grip"' not in webui._render_md("t", raw, cap="c", nonce="n"), "no grip on a raw html cell"
two = "| a | b | c |\n|---|---|---|\n| ☐ | x | KEEP |\n\n| a | b |\n|---|---|\n| ☐ | y |\n"
try:
    e(two, "move", 0, to=1)
    raise AssertionError("moved a row into a table with fewer columns")
except ValueError:
    pass
cr = "a|b\n---|---\n☐|one\r☐|two\n☐|three\n"
assert e(cr, "move", 0, to=2, after=True) == "a|b\n---|---\n☐|two\n☐|three\n☐|one\r", repr(e(cr, "move", 0, to=2, after=True))

# round 3: mixed line endings keep every character; a \\| or a pipe inside a raw html cell
# cannot make a row lose or merge cells
mixed = "a|b\r\n-|-\n☐|one\n☐|two"
assert e(mixed, "move", 0, to=1, after=True) == "a|b\r\n-|-\n☐|two\n☐|one", repr(e(mixed, "move", 0, to=1, after=True))
for src in ("a|b|c\n-|-|-\n☐|x\\\\|KEEP\n\na|b\n-|-\n☐|y\n",
            "<table><tbody><tr><td>\n☐ one | detail\n</td></tr><tr><td>\n☐ two | detail\n</td></tr></tbody></table>"):
    try:
        e(src, "move", 0, to=1, after=True)
        raise AssertionError(f"moved a row that would change: {src!r}")
    except ValueError:
        pass

# round 4: a move must not change anything outside the moved row (lone \r + \n would join)
lone = "a|b\n-|-\r☐|one\n\nKEEP | THIS | TEXT\n\na|b\n-|-\n☐|two\n"
try:
    e(lone, "move", 0, to=1, after=True)
    raise AssertionError("a move changed content outside the row")
except ValueError:
    pass

# round 5: moving a table's last row out; dropping on a header (also of an empty table);
# images in raw html cells are part of the row
last = "a|b\n-|-\n☐|one\n\na|b\n-|-\n☐|two\n"
assert e(last, "move", 0, to=1, after=True) == "a|b\n-|-\n\na|b\n-|-\n☐|two\n☐|one\n"
empty = "a|b\n-|-\n☐|one\n\nc|d\n-|-\n"
assert e(empty, "move", 0, line=5) == "a|b\n-|-\n\nc|d\n-|-\n☐|one\n", repr(e(empty, "move", 0, line=5))
assert e(last, "move", 1, line=1) == "a|b\n-|-\n☐|two\n☐|one\n\na|b\n-|-\n", "header drop = first row"
for bad_line in (0, 2, 99):
    try:
        e(last, "move", 0, line=bad_line)
        raise AssertionError(f"moved under a non-separator line {bad_line}")
    except ValueError:
        pass
hd = webui._render_md("t", empty, cap="c", nonce="n")
assert hd.count('class="hdr"') == 2 and 'data-l="1"' in hd and 'data-l="5"' in hd and "" not in hd, hd
fenced = webui._render_md("t", "- ☐ x\n\n```\na|b\n-|-\n```", cap="c", nonce="n")
assert "" not in fenced and 'class="hdr"' not in fenced, "no header marker leaks from a fence"
img = ("<table><tbody><tr><td>\n☐ | same\n☐ | same\n<img src=\"a.png\">\n</td></tr><tr><td>\n☐ | same\n"
       "<img src=\"b.png\">\n</td></tr></tbody></table>")
try:
    e(img, "move", 0, to=2, after=True)
    raise AssertionError("a raw html move changed which image a box sits with")
except ValueError:
    pass

# round 6: header tags never change how a page renders; raw html rows never move
for src in ("- ☐ x\n\n# heading | text\n---\n", "- ☐ x\n\n<div>one | two\n---|---\nhello\n</div>\n"):
    assert "\ue003" not in webui._html(src), f"header tag would change the render of {src!r}"
assert "<h1" in webui._render_md("t", "- ☐ x\n\n# heading | text\n---\n", cap="c", nonce="n")
assert "\ue003" in webui._html(CL), "ordinary pipe tables keep their header targets"
bold = "<table><tbody><tr><td><b>\n☐ | same</b>\n☐ | same\n</td></tr><tr><td><b>\n☐ | same\n</b></td></tr></tbody></table>"
try:
    e(bold, "move", 0, to=2, after=True)
    raise AssertionError("moved a raw html row")
except ValueError:
    pass

# round 7: one bad header candidate does not cost the other tables; a table on a list-item line
mixed_h = "# Plan | Tasks\n---\n\na|b\n-|-\n☐|one\n☐|two"
assert webui._kinds(mixed_h) == {0: "row", 1: "row"}, webui._kinds(mixed_h)
li_tab = "- | Done | Task |\n  |---|---|\n  | ☐ | one |\n  | ☐ | two |"
assert webui._kinds(li_tab) == {0: "row", 1: "row"}, webui._kinds(li_tab)

# an MCP write racing a checklist edit: the shared lock makes the edit see it (409), never drop it
import threading as _th  # noqa: E402
cap = re.search(r'data-cap="([\w-]+)"', anon.get(c.get("/view?path=todo", follow_redirects=False)
                                               .headers["location"]).text).group(1)
got = {}
with server.page_lock():
    w = _th.Thread(target=lambda: got.update(r=c.post("/api/page/edit", json={**body, "cap": cap})))
    w.start()
    time.sleep(0.3)
    assert w.is_alive(), "the edit must wait for the lock"
    p = server.PAGES / "todo" / "index.md"
    p.write_text(p.read_text() + "\nadded by the MCP\n")
w.join(5)
assert got["r"].status_code == 409 and "added by the MCP" in p.read_text(), got["r"].text

print("OK: all checks pass")
