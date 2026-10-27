"""Checklist pages in a real browser (optional: needs playwright + chromium).
Run with any python that has playwright: python test_browser.py"""
import os
import re
import signal
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
for sub in ("proj", "proj/sub"):  # a folder with a subpage, for the sidebar tree
    (tmp / sub).mkdir()
    (tmp / sub / "index.md").write_text(f"# {sub}\n")
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
here = Path(__file__).resolve().parent
uvicorn = here / ".venv" / "bin" / "uvicorn"
env = {**os.environ, "QILLQA_PAGES_DIR": str(tmp), "QILLQA_TOKEN": "browser-tok"}
srv = subprocess.Popen([str(uvicorn), "webui:app", "--port", str(port)], cwd=here, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
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
        # sidebar: folders start collapsed, the user opens them, the choice survives a reload
        pg.goto(base + "/#plan")
        sub, fold = pg.locator('nav a[data-path="proj/sub"]'), pg.locator('nav a[data-path="proj"]').locator("xpath=../button")
        pg.locator('nav a[data-path="proj"]').wait_for()
        assert sub.is_hidden(), "folders should start collapsed"
        fold.click()
        assert sub.is_visible() and fold.get_attribute("aria-expanded") == "true", "fold did not open"
        pg.reload()
        sub.wait_for()
        assert sub.is_visible(), "opened folder not remembered"
        fold.click()
        assert sub.is_hidden(), "fold did not close"
        pg.goto(base + "/#proj/sub")
        assert "in" in pg.locator('nav a[data-path="proj"]').get_attribute("class"), "collapsed parent not marked"
        pg.click("#side")
        assert pg.locator("aside").is_hidden(), "sidebar did not hide"
        pg.reload()
        assert pg.locator("aside").is_hidden(), "hidden sidebar not remembered"
        assert pg.evaluate("document.querySelector('section').getBoundingClientRect().left") < 5, "page did not take the width"
        pg.click("#side")
        assert pg.locator("aside").is_visible(), "sidebar did not come back"
        pg.set_viewport_size({"width": 390, "height": 800})
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390, "overflow at 390 px"
        pg.set_viewport_size({"width": 1200, "height": 800})
        # search box: folded behind 🔍 until used; the header keeps its height (main is sized from it)
        h0 = pg.evaluate("document.querySelector('header').offsetHeight")
        assert pg.locator("#q").is_hidden() and pg.locator("#find").is_visible(), "search box should start folded"
        pg.click("#find")
        assert pg.locator("#q").is_visible() and pg.evaluate("document.activeElement.id") == "q", "🔍 did not open the box"
        assert pg.evaluate("document.querySelector('header').offsetHeight") == h0, "header height jumps when the box opens"
        pg.keyboard.press("Escape")
        assert pg.locator("#q").is_hidden(), "Esc did not fold the box"
        pg.keyboard.press("/")
        assert pg.locator("#q").is_visible(), "/ did not open the box"
        pg.locator("nav a").first.focus()  # focus leaves an empty box -> it folds
        assert pg.locator("#q").is_hidden(), "empty box did not fold on blur"
        # phone: only the page; the floating ☰ brings back header + pages, picking a page hides them again
        pg.set_viewport_size({"width": 390, "height": 800})
        assert pg.locator("header").is_hidden() and pg.locator("aside").is_hidden(), "phone should show only the page"
        assert pg.evaluate("document.querySelector('#page').offsetHeight") >= 790, "page does not fill the phone"
        pg.click("#menu")
        assert pg.locator("header").is_visible() and pg.locator('nav a[data-path="plan"]').is_visible(), "☰ did not open the bar"
        assert pg.locator("#q").is_visible() and pg.locator("#find").is_hidden(), "phone bar should show the search field at once"
        top = lambda sel: pg.evaluate(f"document.querySelector('{sel}').getBoundingClientRect().top")  # noqa: E731
        assert abs(top(".brand") - top(".out")) < 12 and top("#q") > top(".out"), "brand + sign out on one row, field below"
        assert pg.evaluate("document.querySelector('#q').offsetWidth") > 340, "field does not take the row"
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390, "overflow at 390 px with the bar open"
        pg.click('nav a[data-path="plan"]')
        assert pg.locator("header").is_hidden(), "picking a page did not close the bar"
        pg.set_viewport_size({"width": 1200, "height": 800})
        pg.goto(base + "/#plan")
        fr = pg.frame_locator("#page")
        frame = lambda: pg.frames[1]  # noqa: E731

        def until(js, what, secs=30):
            """Poll the page (it reloads after every edit, so evaluate may hit a dead context)."""
            end = time.time() + secs
            while time.time() < end:
                try:
                    if frame().evaluate(js):
                        return
                except Exception:
                    pass
                time.sleep(0.1)
            raise AssertionError(what)

        cap = lambda: frame().evaluate("document.querySelector('article').dataset.cap")  # noqa: E731

        def rerendered(old, what):
            until(f"document.readyState==='complete' && document.querySelector('article')?.dataset.cap!=='{old}'", what)

        until("document.readyState==='complete' && !!document.querySelector('button.cb')", "page did not load")

        # click: saved to the file, page back at the same scroll
        frame().evaluate("scrollTo(0,200)")
        c0 = cap()
        fr.locator("tr", has_text=re.compile(r"task 2\b")).locator("button.cb").click()
        rerendered(c0, "no reload after the click")
        assert "| ☑ | task 2 |" in md.read_text(), "click not saved"
        until("scrollY > 150", "scroll not restored")

        # finger 7 drags 'task 3' over 'task 5'; finger 9 lifting must not drop it, finger 7 lifting must
        frame().evaluate("scrollTo(0,0)")
        pg.wait_for_timeout(200)
        frame().evaluate("""() => {
          const row = t => [...document.querySelectorAll('tr')].find(r => [...r.cells].some(c => c.textContent.trim() === t));
          const g = row('task 3').querySelector('.grip'), r5 = row('task 5').getBoundingClientRect();
          const ev = (n, id, x, y) => new PointerEvent(n, {pointerId: id, bubbles: true, clientX: x, clientY: y});
          g.dispatchEvent(ev('pointerdown', 7, 5, 5));
          document.dispatchEvent(ev('pointermove', 7, r5.left + 40, r5.top + r5.height * 0.75));}""")
        pg.wait_for_timeout(300)
        frame().evaluate("document.dispatchEvent(new PointerEvent('pointerup',{pointerId:9,bubbles:true}))")
        pg.wait_for_timeout(500)
        assert md.read_text() == PAGE.replace("| ☐ | task 2 |", "| ☑ | task 2 |"), "a foreign pointer dropped the row"
        assert frame().evaluate("document.querySelectorAll('tr.dragging').length") == 1, "drag ended early"
        c0 = cap()
        frame().evaluate("document.dispatchEvent(new PointerEvent('pointerup',{pointerId:7,bubbles:true}))")
        rerendered(c0, "no reload after the owner's drop")
        t = md.read_text()
        assert t.index("task 4 ") < t.index("task 5 ") < t.index("task 3 ") < t.index("task 6 "), "owner drop failed"
        grip = fr.locator("tr", has_text=re.compile(r"task 1\b")).locator(".grip")

        # drag 'task 1' down a long page into Done, before 'first done' (auto-scroll near the edge)
        frame().evaluate("scrollTo(0,0)")
        until("scrollY === 0", "did not scroll to the top")
        c0 = cap()
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
        rerendered(c0, "no reload after the long drag")
        done = md.read_text().split("## Done")[1]
        assert "task 1 " in done and done.index("task 1") < done.index("first done"), "drag not saved"

        # drop on a table's header: the row becomes the first of that table
        c0 = cap()
        frame().evaluate("""() => {
          const th = [...document.querySelectorAll('thead tr')].find(r => r.querySelector('.hdr') &&
                       r.closest('table').textContent.includes('first done'));
          th.scrollIntoView({block: 'center'});
          const row = [...document.querySelectorAll('tr')].find(r => [...r.cells].some(c => c.textContent.trim() === 'task 12'));
          const b = th.getBoundingClientRect(), ev = (n, x, y) => new PointerEvent(n, {pointerId: 11, bubbles: true, clientX: x, clientY: y});
          row.querySelector('.grip').dispatchEvent(ev('pointerdown', b.left + 40, b.top + b.height / 2));  // no edge auto-scroll
          document.dispatchEvent(ev('pointermove', b.left + 40, b.top + b.height / 2));}""")
        until("document.querySelectorAll('thead .drop-after, thead tr.drop-after').length === 1", "no header drop mark")
        frame().evaluate("document.dispatchEvent(new PointerEvent('pointerup',{pointerId:11,bubbles:true}))")
        rerendered(c0, "no reload after the header drop")
        done = md.read_text().split("## Done")[1]
        assert done.index("task 12") < done.index("task 1 "), "header drop did not put the row first"

        # stale page: the file changes behind its back -> 409 -> reload, the click is not applied
        md.write_text(md.read_text().replace("first done", "first done (edited)"))
        before, c0 = md.read_text(), cap()
        fr.locator("button.cb").first.click()
        rerendered(c0, "no reload after 409")
        assert md.read_text() == before, "a stale page overwrote the file"
        assert fr.locator("tr", has_text="(edited)").count() == 1, "no reload after 409"
        b.close()
finally:  # the whole group: an edit reindexes, which spawns the embedder child
    os.killpg(srv.pid, signal.SIGTERM)
print("OK: browser checks pass")
