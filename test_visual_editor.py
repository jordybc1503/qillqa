"""Visual editor (Crepe) on markdown pages, against a throwaway notebook. Needs network (esm.sh, jsdelivr).

  .venv/bin/python test_visual_editor.py
"""
import os, socket, subprocess, tempfile, time, urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

STORY = """# Cuento 1: La prueba

_Un cuento de la sierra_

> Fuente: nota interna.

Killa era una osita de anteojos.

—¡Quiero descubrirlo todo! —decía siempre, «con cariño».

![La plaza gris](img/01.jpg)

Kusi tenía un don especial: sabía escuchar.

## Para jugar después del cuento

¿Por qué nació la Niebla Gris?
"""
TABLE = "# Tabla\n\n| a | b |\n|---|---|\n| ☐ uno | 1 |\n"

tmp = Path(tempfile.mkdtemp())
(tmp / "story/img").mkdir(parents=True)
(tmp / "story/index.md").write_text(STORY)
(tmp / "story/page.css").write_text("article h1{color:rgb(1, 2, 3)}")
JPG = Path.home() / "code/alexa/killa-y-kusi/cuentos/cuento-1-huaraz/img/01.jpg"
(tmp / "story/img/01.jpg").write_bytes(JPG.read_bytes())
(tmp / "table").mkdir()
(tmp / "table/index.md").write_text(TABLE)

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
here = Path(__file__).resolve().parent
env = {**os.environ, "QILLQA_PAGES_DIR": str(tmp), "QILLQA_TOKEN": "vis-tok"}
srv = subprocess.Popen([str(here / ".venv/bin/uvicorn"), "webui:app", "--port", str(port)], cwd=here, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
base = f"http://127.0.0.1:{port}"
story = lambda: (tmp / "story/index.md").read_text()
try:
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    with sync_playwright() as p:
        chrome = os.path.expanduser("~/.local/opt/google-chrome/chrome")
        b = p.chromium.launch(**({"executable_path": chrome} if os.path.exists(chrome) else {}))
        pg = b.new_page(viewport={"width": 1200, "height": 900})
        pg.on("dialog", lambda d: d.accept())
        pg.goto(base + "/login")
        pg.fill("input[name=token]", "vis-tok")
        pg.click("button")

        # page.css styles the reading view
        pg.goto(base + "/#story")
        h1 = pg.frame_locator("#page").locator("h1")
        h1.wait_for()
        assert h1.evaluate("e => getComputedStyle(e).color") == "rgb(1, 2, 3)", "page.css not applied"

        # a story opens in the visual editor, with its image showing
        pg.locator("#crumb a.edl", has_text="Edit").click()
        pg.locator("#ed-visual .ProseMirror").wait_for(timeout=60000)
        assert pg.locator("#ed-text").is_hidden(), "text editor should be hidden in visual mode"
        pg.wait_for_function("[...document.querySelectorAll('#ed-visual img')].some(i => i.naturalWidth > 0)",
                             timeout=15000)

        # untouched save: byte-identical, no display URL leaks into the file
        before = story()
        pg.locator("#ed-save").click()
        pg.locator("#ed-msg", has_text="Saved").wait_for(timeout=8000)
        assert story() == before, "an untouched save changed the file"

        # type at the end of one paragraph: exactly one line changes
        pg.locator("#ed-visual p", has_text="sabía escuchar").click()
        pg.keyboard.press("End")
        pg.keyboard.type(" Y también cantaba.")
        pg.keyboard.press("Control+s")
        pg.locator("#ed-msg", has_text="Saved").wait_for(timeout=8000)
        a, c = before.splitlines(), story().splitlines()
        diff = [(x, y) for x, y in zip(a, c) if x != y]
        assert len(a) == len(c) and diff == [("Kusi tenía un don especial: sabía escuchar.",
                                             "Kusi tenía un don especial: sabía escuchar. Y también cantaba.")], diff
        assert "/f/" not in story(), "a signed URL was saved"

        # no keyboard shortcut needed: selecting text shows the formatting bar (phones)
        pg.locator("#ed-visual p", has_text="sabía escuchar").dblclick()
        bar = pg.locator(".milkdown-toolbar")
        bar.wait_for(timeout=5000)
        assert bar.locator(".toolbar-item").count() >= 3, "formatting bar without buttons"
        assert bar.locator(".toolbar-item").count() == 4, bar.inner_html()[:300]  # bold, italic, code, link: no ~~
        pg.keyboard.press("Escape")

        # bold + a line break (Shift+Enter) render as such in the view
        pg.locator("#ed-visual p", has_text="Killa era una osita").click()
        pg.keyboard.press("End")
        pg.keyboard.type(" Era ")
        pg.keyboard.press("Control+b")
        pg.keyboard.type("valiente")
        pg.keyboard.press("Control+b")
        pg.keyboard.press("Shift+Enter")
        pg.keyboard.type("y curiosa.")
        pg.keyboard.press("Control+s")
        pg.locator("#ed-msg", has_text="Saved").wait_for(timeout=8000)
        assert "**valiente**  \ny curiosa." in story(), repr(story()[:300])
        assert "\\\n" not in story() and " " not in story()
        pg.locator("#crumb a.edl", has_text="View").click()
        art = pg.frame_locator("#page").locator("article")
        art.locator("strong", has_text="valiente").wait_for()
        assert art.locator("p", has_text="valiente").locator("br").count() == 1, "line break not rendered"

        # a page the visual editor would change stays in the text editor
        pg.goto(base + "/#table")
        pg.locator("#crumb a.edl", has_text="Edit").click()
        pg.locator("#ed-msg", has_text="text mode").wait_for(timeout=60000)
        assert pg.locator("#ed-text").is_visible() and pg.locator("#ed-visual").is_hidden()
        assert (tmp / "table/index.md").read_text() == TABLE
        pg.locator("#ed-cancel").click()

        # switch to text and back; the choice is remembered
        pg.goto(base + "/#story")
        pg.locator("#crumb a.edl", has_text="Edit").click()
        pg.locator("#ed-visual .ProseMirror").wait_for(timeout=60000)
        pg.locator("#ed-mode").click()
        assert pg.locator("#ed-text").is_visible() and "valiente" in pg.locator("#ed-text").input_value()
        pg.locator("#ed-mode").click()
        pg.locator("#ed-visual .ProseMirror").wait_for()

        # phone width: the editor fits
        pg.set_viewport_size({"width": 390, "height": 844})
        w = pg.evaluate("document.querySelector('#ed-visual').getBoundingClientRect().width")
        assert w <= 390, w
        pg.screenshot(path=os.environ.get("SHOT", str(tmp / "phone.png")))
        b.close()
    print("visual editor: all checks passed")
finally:
    os.killpg(srv.pid, 15)
