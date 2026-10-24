"""Checklist pages in a real browser (optional: needs playwright + chromium).
Run with any python that has playwright: python test_browser.py"""
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("SKIP: playwright is not installed")
    sys.exit(0)

ROWS = "\n".join(f"| ☐ | task {n} | detail |" for n in range(1, 13))
PAGE = f"""# Plan

| ✓ | Task | Detail |
|---|---|---|
{ROWS}

{"filler paragraph. " * 400}

## Done

| ✓ | Task | Detail |
|---|---|---|
| ☑ | first done | x |
"""
tmp = Path(tempfile.mkdtemp())
(tmp / "plan").mkdir()
md = tmp / "plan" / "index.md"
md.write_text(PAGE)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
here = Path(__file__).resolve().parent
uvicorn = here / ".venv" / "bin" / "uvicorn"
env = {**os.environ, "QILLQA_PAGES_DIR": str(tmp), "QILLQA_TOKEN": "browser-tok"}
srv = subprocess.Popen([str(uvicorn), "webui:app", "--port", str(port)], cwd=here, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = f"http://127.0.0.1:{port}"
try:
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1200, "height": 800})
        pg.goto(base + "/login")
        pg.fill("input[name=token]", "browser-tok")
        pg.click("button")
        pg.goto(base + "/#plan")
        pg.wait_for_timeout(1000)
        fr = pg.frame_locator("#page")
        frame = lambda: pg.frames[1]  # noqa: E731

        # click: saved to the file, page back at the same scroll
        frame().evaluate("scrollTo(0,200)")
        fr.locator("tr", has_text=re.compile(r"task 2\b")).locator("button.cb").click()
        pg.wait_for_timeout(1200)
        assert "| ☑ | task 2 |" in md.read_text(), "click not saved"
        assert frame().evaluate("scrollY") > 150, "scroll not restored"

        # a second finger lifting does not drop the row being dragged
        frame().evaluate("scrollTo(0,0)")
        grip = fr.locator("tr", has_text=re.compile(r"task 1\b")).locator(".grip")
        frame().evaluate("""() => {const g=document.querySelector('tr .grip');
          g.dispatchEvent(new PointerEvent('pointerdown',{pointerId:7,bubbles:true,clientX:5,clientY:5}));
          document.dispatchEvent(new PointerEvent('pointerup',{pointerId:9,bubbles:true}));
          document.dispatchEvent(new PointerEvent('pointercancel',{pointerId:7,bubbles:true}));}""")
        pg.wait_for_timeout(300)
        assert md.read_text().index("task 1") < md.read_text().index("## Done"), "a foreign pointer moved the row"

        # drag 'task 1' down a long page into Done, before 'first done' (auto-scroll near the edge)
        frame().evaluate("scrollTo(0,0)")
        bb = grip.bounding_box()
        pg.mouse.move(bb["x"] + 5, bb["y"] + 8)
        pg.mouse.down()
        pg.mouse.move(bb["x"] + 5, 790, steps=5)
        for k in range(150):
            tb = fr.locator("tr", has_text="first done").bounding_box()
            if tb and 120 < tb["y"] and tb["y"] + tb["height"] < 740:  # fully visible
                break
            pg.mouse.move(bb["x"] + 5, 792 if k % 2 else 789)
            pg.wait_for_timeout(50)
        tb = fr.locator("tr", has_text="first done").bounding_box()
        pg.mouse.move(tb["x"] + 40, tb["y"] + 5, steps=4)
        assert frame().evaluate("document.querySelectorAll('.drop-before').length") == 1, "no drop mark"
        pg.mouse.up()
        pg.wait_for_timeout(1200)
        done = md.read_text().split("## Done")[1]
        assert "task 1 " in done and done.index("task 1") < done.index("first done"), "drag not saved"

        # stale page: the file changes behind its back -> 409 -> reload, the click is not applied
        md.write_text(md.read_text().replace("first done", "first done (edited)"))
        before = md.read_text()
        fr.locator("button.cb").first.click()
        pg.wait_for_timeout(1200)
        assert md.read_text() == before, "a stale page overwrote the file"
        assert fr.locator("tr", has_text="(edited)").count() == 1, "no reload after 409"
        b.close()
finally:
    srv.terminate()
print("OK: browser checks pass")
