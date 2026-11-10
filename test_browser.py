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
(tmp / "bare" / "kid").mkdir(parents=True)  # a folder with no page of its own
(tmp / "bare" / "kid" / "index.md").write_text("# kid\n")
now = time.time()  # last changed: bare (now) > proj (its subpage, 1 h ago) > plan (10 days ago)
for f, age in (("plan", 10 * 86400), ("proj", 2 * 86400), ("proj/sub", 3600), ("bare/kid", 0)):
    os.utime(tmp / f / "index.md", (now - age, now - age))
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
        chrome = os.path.expanduser(os.environ.get("QILLQA_CHROME", "~/.local/opt/google-chrome/chrome"))
        b = p.chromium.launch(**({"executable_path": chrome} if os.path.exists(chrome) else {}))
        pg = b.new_page(viewport={"width": 1200, "height": 800})
        pg.goto(base + "/login")
        pg.fill("input[name=token]", "browser-tok")
        pg.click("button")
        # header auto-hide (default on): seen at load, gone while reading, back from the strip at the top or '/'
        pg.goto(base + "/#plan")
        pg.mouse.move(600, 500, steps=15)
        assert pg.locator("header").is_visible(), "header should show at load"
        pg.locator("header").wait_for(state="hidden", timeout=5000)
        assert pg.evaluate("document.querySelector('main').offsetHeight") >= 795, "page did not take the height"
        pg.mouse.move(600, 3)
        pg.locator("header").wait_for(state="visible", timeout=2000)
        pg.mouse.move(600, 500, steps=15)  # like a real mouse: one synthetic jump into the iframe sends the shell no mouseleave
        pg.locator("header").wait_for(state="hidden", timeout=5000)
        pg.keyboard.press("/")
        assert pg.locator("header").is_visible() and pg.evaluate("document.activeElement.id") == "q", "/ did not bring the search"
        pg.keyboard.type("battery")
        pg.wait_for_timeout(1500)
        assert pg.locator("header").is_visible(), "header hid while the search was in use"
        pg.keyboard.press("Escape")
        pg.locator("header").wait_for(state="hidden", timeout=5000)
        pg.mouse.move(600, 3)
        pg.click("#avatar")
        pg.click("#settings label:has(#autohide)")  # off: the header stays, and that is remembered
        pg.mouse.move(600, 500, steps=15)
        pg.mouse.click(140, 760)  # outside: empty sidebar
        pg.wait_for_timeout(1500)
        assert pg.locator("header").is_visible(), "header hid with auto-hide off"
        pg.reload()
        pg.wait_for_timeout(1800)
        assert pg.locator("header").is_visible(), "auto-hide off not remembered"
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
        # search box: always visible; '/' focuses it; the folder filter shows only while it is in use
        active = lambda: pg.evaluate("document.activeElement.id")  # noqa: E731
        assert pg.locator("#q").is_visible() and pg.locator("#scope").is_hidden(), "search box should always be there"
        pg.keyboard.press("/")
        assert active() == "q" and pg.locator("#scope").is_visible(), "/ did not focus the box"
        pg.keyboard.press("Escape")
        assert active() != "q" and pg.locator("#q").is_visible() and pg.locator("#scope").is_hidden(), "Esc did not leave the box"
        # avatar -> settings: theme applies to the shell AND the sandboxed page, and survives a reload
        pg.goto(base + "/#proj")
        art_bg = lambda: pg.frames[1].evaluate("getComputedStyle(document.querySelector('article')).backgroundColor")  # noqa: E731

        def wait_bg(want, what):
            end = time.time() + 15
            while time.time() < end:
                try:
                    if art_bg() == want:
                        return
                except Exception:
                    pass
                time.sleep(0.1)
            raise AssertionError(f"{what}: {art_bg()!r}")

        wait_bg("rgb(255, 255, 255)", "page not white by default")
        assert pg.locator("#settings").is_hidden(), "settings should start closed"
        pg.click("#avatar")
        assert pg.locator("#settings").is_visible() and pg.get_attribute("#avatar", "aria-expanded") == "true", "avatar did not open settings"
        pg.click("#settings label:has([value=night])")
        assert pg.evaluate("document.documentElement.dataset.theme") == "night", "shell not in night"
        wait_bg("rgb(19, 17, 14)", "page iframe did not switch to night")
        pg.keyboard.press("Escape")
        assert pg.locator("#settings").is_hidden(), "Esc did not close settings"
        pg.reload()
        assert pg.evaluate("document.documentElement.dataset.theme") == "night", "night not remembered"
        wait_bg("rgb(19, 17, 14)", "page iframe not night after reload")
        pg.click("#avatar")
        assert pg.locator("#settings [value=night]").is_checked(), "night radio not checked after reload"
        pg.click("#settings label:has([value=light])")
        wait_bg("rgb(255, 255, 255)", "back to white failed")
        pg.mouse.click(140, 760)  # outside: empty sidebar
        assert pg.locator("#settings").is_hidden(), "outside click did not close settings"
        pg.click("#avatar")
        pg.focus("#settings a")
        pg.keyboard.press("Tab")  # focus leaves the popover -> it closes
        assert pg.locator("#settings").is_hidden() and pg.get_attribute("#avatar", "aria-expanded") == "false", "Tab out did not close"
        pg.click("#avatar")
        pg.keyboard.press("/")  # search opens; Escape there closes the popover too
        pg.keyboard.press("Escape")
        assert pg.locator("#settings").is_hidden(), "Escape outside the popover did not close it"
        # the logo opens the project cards, last changed first; sort by name is remembered
        pg.click(".brand")
        pg.locator(".card").first.wait_for()
        titles = lambda: pg.locator(".card b").all_inner_texts()  # noqa: E731
        assert pg.locator("#page").is_hidden() and titles() == ["bare", "proj", "Plan"], titles()
        proj = pg.locator(".card", has_text="proj").first
        assert "2 pages" in proj.inner_text() and "Last: proj/sub" in proj.inner_text(), proj.inner_text()
        pg.select_option("#sort", "name")
        assert titles() == ["bare", "Plan", "proj"], titles()
        pg.reload()
        pg.locator(".card").first.wait_for()
        assert pg.locator("#sort").input_value() == "name" and titles() == ["bare", "Plan", "proj"], "sort not remembered"
        pg.select_option("#sort", "date")
        proj.locator(".last").click()  # the "Last:" line opens that page
        pg.wait_for_function("location.hash === '#proj/sub'")
        pg.locator("#page").wait_for()
        assert pg.locator("#page").is_visible() and pg.locator("#cards").is_hidden(), "card did not open the page"
        pg.goto(base + "/#bare")  # folder without a page: its subpages as cards, no iframe (its 404 would be unthemed JSON)
        pg.locator(".card").first.wait_for()
        assert pg.locator("#page").is_hidden() and titles() == ["kid"] and pg.inner_text("#cards-title") == "bare", titles()
        pg.goto(base + "/#proj")
        pg.locator("#page").wait_for()
        # breadcrumb: every parent is a link, the current page is not; "Projects" goes back to the cards
        pg.goto(base + "/#proj/sub")
        pg.locator("#crumb a").first.wait_for()
        crumb_text = pg.evaluate("[...document.querySelector('#crumb').childNodes].filter(n=>!(n.classList&&n.classList.contains('exps'))).map(n=>n.textContent).join('')")
        assert crumb_text == "Projects / proj / sub", crumb_text
        assert pg.get_attribute("#crumb a.exp", "href") == "/api/export?path=proj%2Fsub", "export link"
        assert pg.locator("#crumb a.exp").count() == 1, "a page with no subpages offers no '+ subpages'"
        assert pg.locator("#crumb > a").all_inner_texts() == ["Projects", "proj"], pg.locator("#crumb > a").all_inner_texts()
        pg.click("#crumb a:text-is('proj')")
        pg.wait_for_function("[...document.querySelector('#crumb').childNodes].filter(n=>!(n.classList&&n.classList.contains('exps'))).map(n=>n.textContent).join('') === 'Projects / proj'")
        assert pg.locator("#crumb a.exp").all_inner_texts() == ["⤓ Export", "⤓ + subpages"], "page with a subpage"
        pg.click("#crumb a:text-is('Projects')")
        pg.locator(".card").first.wait_for()
        assert pg.inner_text("#cards-title") == "Projects" and pg.inner_text("#crumb") == "", "Projects did not go back to the cards"
        pg.goto(base + "/#proj")
        pg.locator("#page").wait_for()
        # phone: only the page; the floating ☰ brings back header + pages, picking a page hides them again
        pg.set_viewport_size({"width": 390, "height": 800})
        assert pg.locator("header").is_hidden() and pg.locator("aside").is_hidden(), "phone should show only the page"
        assert pg.evaluate("document.querySelector('#page').offsetHeight") >= 790, "page does not fill the phone"
        pg.click("#menu")
        assert pg.locator("header").is_visible() and pg.locator('nav a[data-path="plan"]').is_visible(), "☰ did not open the bar"
        assert pg.locator("#q").is_visible(), "phone bar should show the search field at once"
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

        # export: a page + subpages as one .html, opened from disk with the network cut off
        (tmp / "share" / "sub").mkdir(parents=True)
        (tmp / "share" / "index.md").write_text("# Share\n\n![dot](dot.png)\n\n[to the subpage](#share/sub)\n\n"
                                               '<img src="nope.png" onerror="parent.document.title=\'pwned\'">\n\n'
                                               "[to sec](#sec)\n\n## Sec {#sec}\n")
        dot = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                            "1f15c4890000000d49444154789c63f8cfc0f01f0005000201e5274e"
                            "c70000000049454e44ae426082")
        (tmp / "share" / "dot.png").write_bytes(dot)
        (tmp / "share" / "sub" / "index.md").write_text("# Sub\n\nhello from the subpage\n")
        # an html page: unquoted src, a stylesheet with url(), a page link with no target, an anchor
        (tmp / "share" / "sub" / "app").mkdir()
        (tmp / "share" / "sub" / "app" / "dot.png").write_bytes(dot)
        (tmp / "share" / "sub" / "app" / "style.css").write_text(  # imports a sheet that sorts after it
            '@import url("z theme.css") screen;\n#bg{width:10px;height:10px;background:url(dot.png)}')
        (tmp / "share" / "sub" / "app" / "z theme.css").write_text(
            "#bg2{width:10px;height:10px;background:url( 'dot.png' )}")
        (tmp / "share" / "sub" / "app" / "index.html").write_text(
            "<!doctype html><html><head><title>App</title><link rel=stylesheet href=style.css></head><body>"
            '<img id=u src=dot.png><div id=bg></div><div id=bg2></div><a id=back href="/#share">back</a> '
            '<a id=jump href="#far">jump</a><template id=tp><img id=ti src=dot.png></template>'
            '<script>window.S="<img src=dot.png>";document.body.append(tp.content.cloneNode(true));'
            'const s2=document.createElement("script");s2.innerHTML=\'window.T="<img src=dot.png>"\';document.body.append(s2)</script>'
            '<h2 id=far style="margin-top:3000px">Far</h2></body></html>')
        got = pg.request.get(base + "/api/export?path=share&subpages=1")
        assert got.ok and got.headers["content-disposition"].startswith("attachment"), got.headers
        out = tmp / "share.html"
        out.write_bytes(got.body())
        off = b.new_page(viewport={"width": 1000, "height": 700})
        errs = []
        off.on("console", lambda m: m.type == "error" and not re.search(r"Blocked script execution|ERR_FILE_NOT_FOUND", m.text) and errs.append(m.text))  # the sandbox blocking the onerror test, and its missing nope.png
        off.route("http*://**", lambda r: r.abort())
        off.goto(out.as_uri())
        view = off.frame_locator("#view")
        view.locator("h1", has_text="Share").wait_for(timeout=10000)
        assert view.locator("img[alt=dot]").evaluate("i=>i.decode().then(()=>i.naturalWidth)") == 1, "embedded image"
        assert off.locator("#list a").count() == 3, "page list"
        off.wait_for_timeout(500)
        assert off.title() != "pwned", "a script in a markdown page ran in the export"
        assert view.locator("a", has_text="to sec").get_attribute("href") == "#sec", "heading link kept"
        view.locator("a", has_text="to the subpage").click()
        view.locator("text=hello from the subpage").wait_for(timeout=10000)
        assert off.locator("#list a.on").inner_text() == "Sub" and not errs, errs
        off.locator("#list a", has_text="App").click()
        view.locator("#u").wait_for(timeout=10000)
        assert view.locator("#u").evaluate("i=>i.decode().then(()=>i.naturalWidth)") == 1, "unquoted src"
        assert "blob:" in view.locator("#bg").evaluate("e=>getComputedStyle(e).backgroundImage"), "css url()"
        assert "blob:" in view.locator("#bg2").evaluate("e=>getComputedStyle(e).backgroundImage"), "css @import order"
        assert view.locator("#u").evaluate("()=>window.S") == "<img src=dot.png>", "script text was rewritten"
        assert view.locator("#u").evaluate("()=>window.T") == "<img src=dot.png>", "script.innerHTML was rewritten"
        assert view.locator("#ti").evaluate("i=>i.decode().then(()=>i.naturalWidth)") == 1, "template content"
        assert view.locator("#u").evaluate("()=>document.compatMode") == "CSS1Compat", "doctype kept"
        view.locator("#jump").click()
        off.wait_for_timeout(300)
        assert off.locator("#list a.on").inner_text() == "App", "an #anchor left the page"
        view.locator("#back").click()
        view.locator("h1", has_text="Share").wait_for(timeout=10000)
        assert off.locator("#list a.on").inner_text() == "Share" and not errs, errs
        # editor: ✎ Edit shows the raw source and the folder's files; save, stale save, unsaved guard
        pg.set_viewport_size({"width": 1200, "height": 800})
        (tmp / "edme" / "assets").mkdir(parents=True)
        (tmp / "edme" / "index.md").write_text("# Edme\n\nfirst text\n")
        (tmp / "edme" / "notes.txt").write_text("n")
        (tmp / "edme" / "assets" / "a.json").write_text("{}")
        pg.goto(base + "/")
        pg.reload()  # the tree must know the new page
        pg.goto(base + "/#edme")
        pg.locator("#crumb a.edl", has_text="Edit").click()
        pg.locator("#edit").wait_for()
        assert pg.locator("#page").is_hidden() and pg.input_value("#ed-text") == "# Edme\n\nfirst text\n"
        assert pg.locator("#ed-list a").all_inner_texts() == ["notes.txt", "assets/a.json"], pg.locator("#ed-list a").all_inner_texts()
        pg.fill("#ed-text", "# Edme\n\nsaved from the web\n")
        pg.keyboard.press("Control+s")
        pg.locator("#ed-msg", has_text="Saved").wait_for(timeout=5000)
        assert (tmp / "edme" / "index.md").read_text() == "# Edme\n\nsaved from the web\n"
        (tmp / "edme" / "index.md").write_text("# Edme\n\nchanged by the MCP\n")
        pg.fill("#ed-text", "# Edme\n\nmine\n")
        pg.click("#ed-save")
        pg.locator("#ed-msg", has_text="changed since").wait_for(timeout=5000)
        assert (tmp / "edme" / "index.md").read_text() == "# Edme\n\nchanged by the MCP\n", "stale save overwrote"
        pg.once("dialog", lambda d: d.dismiss())  # unsaved text: leaving asks, and 'cancel' keeps the editor
        pg.evaluate("location.hash='plan'")
        pg.wait_for_timeout(500)
        assert pg.evaluate("location.hash") == "#edme" and pg.locator("#edit").is_visible(), "unsaved text lost"
        seen = []
        pg.once("dialog", lambda d: (seen.append(d.message), d.accept()))
        pg.locator("#crumb a.edl", has_text="View").click()
        pg.frame_locator("#page").locator("text=changed by the MCP").wait_for(timeout=5000)
        assert seen and pg.locator("#edit").is_hidden() and pg.locator("#page").is_visible(), seen
        # Codex r1: a late reply of an older open/save never lands on the editor now open
        pg.evaluate("""()=>{const f=window.fetch;let n=0;window.__slow=u=>String(u).startsWith('/api/source')&&n++===0;
          window.fetch=(u,o)=>window.__slow(u)?new Promise(r=>setTimeout(()=>r(f(u,o)),1200)):f(u,o)}""")
        pg.evaluate("openEdit('edme');openEdit('edme')")  # double click: the first reply comes last
        pg.locator("#edit").wait_for()
        pg.fill("#ed-text", "typed meanwhile")
        pg.wait_for_timeout(1600)
        assert pg.input_value("#ed-text") == "typed meanwhile", "a stale open replaced unsaved text"
        (tmp / "other" / "index.md").parent.mkdir()
        (tmp / "other" / "index.md").write_text("# Other\n")
        pg.evaluate("""()=>{let n=0;const f=window.fetch;window.fetch=(u,o)=>o&&o.method==='POST'&&n++===0?
          new Promise(r=>setTimeout(()=>r(f(u,o)),1200)):f(u,o)}""")
        pg.evaluate("void saveEdit()")  # not awaited: slow save of edme, then the user opens another page
        pg.evaluate("closeEdit(true);location.hash='other'")
        pg.wait_for_timeout(300)
        pg.evaluate("openEdit('other')")
        pg.locator("#ed-name").wait_for()
        pg.wait_for_function("editing==='other'")
        ver = pg.evaluate("edVer")
        pg.wait_for_timeout(1500)
        assert pg.evaluate("[editing,edVer,dirty()]") == ["other", ver, False], "edme's save touched other's editor"
        assert (tmp / "edme" / "index.md").read_text() == "typed meanwhile", "the slow save itself must still land"
        # Codex r2: a slow save of one page never blocks saving the page opened after it
        # the first POST is held until the second save is done: deterministic, no timing
        pg.evaluate("""()=>{let n=0;const f=window.fetch,gate=new Promise(r=>window.__go=r);
          window.fetch=(u,o)=>o&&o.method==='POST'&&n++===0?gate.then(()=>f(u,o)):f(u,o)}""")
        pg.fill("#ed-text", "# Other\n\nedited\n")
        pg.evaluate("void saveEdit()")  # other: slow
        pg.evaluate("closeEdit(true);location.hash='edme'")
        pg.wait_for_timeout(300)
        pg.evaluate("openEdit('edme')")
        pg.wait_for_function("editing==='edme'")
        pg.fill("#ed-text", "edme again")
        pg.evaluate("void saveEdit()")  # must not be swallowed by other's pending save
        pg.locator("#ed-msg", has_text="Saved").wait_for(timeout=5000)
        assert (tmp / "edme" / "index.md").read_text() == "edme again"
        pg.evaluate("window.__go()")  # now let other's save through
        for _ in range(50):
            if (tmp / "other" / "index.md").read_text() == "# Other\n\nedited\n":
                break
            pg.wait_for_timeout(100)
        assert (tmp / "other" / "index.md").read_text() == "# Other\n\nedited\n"
        # share dialog: a clipboard reply that lands after you moved to another page changes nothing
        pg.evaluate("""()=>{window.__rej=null;Object.defineProperty(navigator.clipboard,'writeText',
          {configurable:true,value:()=>new Promise((_,rej)=>{window.__rej=rej})})}""")
        pg.evaluate("openShare('plan')")
        pg.click("#sh-create")
        pg.wait_for_function("!!window.__rej")  # (a bare function would be CALLED as the predicate)
        assert pg.input_value("#sh-url").startswith(base + "/s/")
        pg.evaluate("document.getElementById('share').close()")
        pg.evaluate("openShare('proj')")
        pg.evaluate("window.__rej(new Error('denied'))")
        pg.wait_for_timeout(300)
        assert pg.input_value("#sh-url") == "" and pg.is_hidden("#sh-out"), "a late reply put another page's link"
        b.close()
finally:  # the whole group: an edit reindexes, which spawns the embedder child
    os.killpg(srv.pid, signal.SIGTERM)
print("OK: browser checks pass")
