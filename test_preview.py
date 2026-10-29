"""preview_page and its MCP App: .venv/bin/python test_preview.py
The screenshot part needs playwright (and Chrome/Chromium); it skips without it."""
import asyncio
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import struct
import time
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
(tmp / "pages").mkdir()
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
PORT = sock.getsockname()[1]
sock.close()
os.environ.update(QILLQA_PAGES_DIR=str(tmp / "pages"), QILLQA_TOKEN="t0k-preview", QILLQA_PORT=str(PORT),
                  QILLQA_VAR_DIR=str(tmp / "var"))  # never touch the real server's pid file

import server  # noqa: E402

UI = "ui://qillqa/preview"

# The tool points MCP Apps hosts at the UI resource; the resource has the MCP Apps type.
tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
assert tools["preview_page"].meta == {"ui": {"resourceUri": UI}}, tools["preview_page"].meta
listed = [r for r in asyncio.run(server.mcp.list_resources()) if str(r.uri) == UI]
assert listed and listed[0].mimeType == "text/html;profile=mcp-app", listed
html = list(asyncio.run(server.mcp.read_resource(UI)))[0]
assert html.mime_type == "text/html;profile=mcp-app"
for needle in ("ui/initialize", "ui/notifications/initialized", "ui/notifications/tool-result",
               "2026-01-26", "e.source !== parent"):
    assert needle in html.content, needle

# Bad input answers isError, without a browser.
r = asyncio.run(server.preview_page("nope"))
assert r.isError and "no page" in r.content[0].text, r
r = asyncio.run(server.preview_page("../etc", width=1100))
assert r.isError, r
r = asyncio.run(server.preview_page("nope", width=50))
assert r.isError and "width" in r.content[0].text, r

try:
    import playwright  # noqa: F401
except ImportError:
    print("SKIP screenshot: playwright is not installed")
    raise SystemExit(0)


def png_size(b64: str) -> tuple[int, int]:
    import base64
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", raw[16:24])


try:
    server.write_page("demo", "# Demo\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", "md", create=True)
    server.write_page("wide", '<!doctype html><title>Wide</title><div style="width:2000px">x</div>'
                              '<script>document.title="ran"</script>', "html", create=True)

    r = asyncio.run(server.preview_page("demo"))
    assert not r.isError, r.content
    img, txt = r.content
    assert img.type == "image" and img.mimeType == "image/png"
    w, h = png_size(img.data)
    info = json.loads(txt.text)
    assert w == 1100 and h >= 100, (w, h)
    assert info["title"] == "Demo" and info["horizontal_overflow"] is False, info
    assert r.structuredContent == info and "data" not in json.dumps(info)  # no image in the UI payload

    r = asyncio.run(server.preview_page("wide", width=390))
    w, _ = png_size(r.content[0].data)
    info = json.loads(r.content[1].text)
    assert w > 390 and info["horizontal_overflow"] is True, (w, info)  # full page = the overflow too

    # Pages taller than the cap come back cut AT the cap (not at the 900 px viewport).
    server.write_page("tall", '<!doctype html><title>Tall</title><div style="height:9000px">t</div>', "html", create=True)
    r = asyncio.run(server.preview_page("tall"))
    assert png_size(r.content[0].data)[1] == server.MAX_PREVIEW_HEIGHT and json.loads(r.content[1].text)["cut_at"]

    # Page JS run by preview_page cannot reach other local services (only its own origin and the CDNs).
    hits = []

    class Spy(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append(self.path)
            self.send_response(204)
            self.end_headers()
        do_GET = do_POST

        def log_message(self, *a):
            pass

    spy = HTTPServer(("127.0.0.1", 0), Spy)
    threading.Thread(target=spy.serve_forever, daemon=True).start()
    server.write_page("ssrf", f'<!doctype html><title>x</title><img src="http://127.0.0.1:{spy.server_port}/img">'
                              f'<script>fetch("http://127.0.0.1:{spy.server_port}/send",'
                              '{method:"POST",mode:"no-cors",body:"hi"})</script>', "html", create=True)
    r = asyncio.run(server.preview_page("ssrf"))
    time.sleep(0.5)
    spy.shutdown()
    assert not r.isError and hits == [], hits
    # The MCP App, inside a mock host that speaks the ext-apps protocol.
    from playwright.sync_api import sync_playwright

    result = asyncio.run(server.preview_page("demo")).model_dump(by_alias=True, exclude_none=True)
    host = """<iframe id="app" style="width:600px;height:400px"></iframe><script>
    window.got = [];
    const app = document.getElementById("app");
    addEventListener("message", e => {
      if (e.source !== app.contentWindow) return;
      const m = e.data; window.got.push(m.method || "result");
      if (m.method === "ui/initialize") app.contentWindow.postMessage({jsonrpc: "2.0", id: m.id,
        result: {protocolVersion: "2026-01-26", hostInfo: {name: "mock", version: "1"}, hostCapabilities: {}}}, "*");
      if (m.method === "ui/notifications/initialized") app.contentWindow.postMessage({jsonrpc: "2.0",
        method: "ui/notifications/tool-result", params: __RESULT__}, "*");
      if (m.method === "ui/open-link") { window.opened = m.params.url;
        app.contentWindow.postMessage({jsonrpc: "2.0", id: m.id, result: {}}, "*"); }
    });
    app.srcdoc = __UI__;
    </script>""".replace("__UI__", json.dumps(html.content).replace("<", "\\u003c")).replace("__RESULT__", json.dumps(result))
    with sync_playwright() as p:
        kw = {"executable_path": server.CHROME} if os.path.exists(server.CHROME) else {}
        b = p.chromium.launch(args=["--no-sandbox"], **kw)
        pg = b.new_page()
        pg.set_content(host)
        frame = pg.frame_locator("#app")
        frame.locator("#open").click()  # visible only once the tool result arrived
        pg.wait_for_function("window.opened")
        assert pg.evaluate("window.opened") == result["structuredContent"]["url"]
        assert frame.locator("#img").evaluate("i => i.naturalWidth") == 1100
        assert frame.locator("#t").inner_text() == "Demo"
        got = pg.evaluate("window.got")
        assert got[:2] == ["ui/initialize", "ui/notifications/initialized"], got
        assert "ui/notifications/size-changed" in got, got
        b.close()
    # show_page: tool/resource metadata, the app-only snapshot tool, then the live view
    # in a mock host that applies the spec's default CSP and serves tools/call.
    LIVE = "ui://qillqa/live"
    assert tools["show_page"].meta == {"ui": {"resourceUri": LIVE}}
    assert tools["page_snapshot"].meta == {"ui": {"resourceUri": LIVE, "visibility": ["app"]}}
    live = list(asyncio.run(server.mcp.read_resource(LIVE)))[0]
    assert live.mime_type == "text/html;profile=mcp-app" and "page_snapshot" in live.content
    (tmp / "pages" / "wide" / "app.js").write_text('document.body.insertAdjacentHTML("beforeend","<p id=js>js ran</p>")')
    (tmp / "pages" / "wide" / "dot.png").write_bytes(__import__("base64").b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
    server.write_page("wide", '<!doctype html><title>Wide</title><p id=v>one</p><img id=dot src="dot.png">'
                              '<script src="app.js"></script>', "html", create=False)
    shown = asyncio.run(server.show_page("wide"))
    s = shown.structuredContent
    page_html = shown.meta["qillqa/html"]
    assert not shown.isError and "html" not in s and "<p" not in shown.content[0].text, shown  # HTML only in _meta
    assert "insertAdjacentHTML" in page_html and 'src="data:image/png;base64,' in page_html, page_html[:300]
    same = asyncio.run(server.page_snapshot("wide", s["version"])).structuredContent
    assert same == {"changed": False, "version": s["version"]}, same

    csp = ('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'self\' '
           '\'unsafe-inline\'; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data:; media-src \'self\' data:; '
           'connect-src \'none\'">')  # the host's default CSP for a View (spec)
    live_doc = live.content.replace("<head>", "<head>" + csp, 1)

    def live_host(first) -> str:
        return """<iframe id="app" style="width:700px;height:500px"></iframe><script>
        window.got = [];
        const app = document.getElementById("app");
        const reply = (id, result) => app.contentWindow.postMessage({jsonrpc: "2.0", id, result}, "*");
        addEventListener("message", async e => {
          if (e.source !== app.contentWindow) return;
          const m = e.data; window.got.push(m.method || "result");
          if (m.method === "ui/initialize") reply(m.id, {protocolVersion: "2026-01-26", hostCapabilities: {}});
          if (m.method === "ui/notifications/initialized") app.contentWindow.postMessage({jsonrpc: "2.0",
            method: "ui/notifications/tool-result", params: __RESULT__}, "*");
          if (m.method === "tools/call") reply(m.id, await window.snap(m.params.arguments));
        });
        app.srcdoc = __UI__;
        </script>""".replace("__UI__", json.dumps(live_doc).replace("<", "\\u003c")).replace(
            "__RESULT__", json.dumps(first.model_dump(by_alias=True, exclude_none=True)).replace("<", "\\u003c"))

    server.write_page("mdx", "# Md\n\n<p id=m>md</p>\n\n<script>document.body.insertAdjacentHTML("
                             "'beforeend','<p id=bad>ran</p>')</script>\n", "md", create=True)
    md_first = asyncio.run(server.show_page("mdx"))  # asyncio.run can't run inside playwright's loop
    with sync_playwright() as p:
        kw = {"executable_path": server.CHROME} if os.path.exists(server.CHROME) else {}
        b = p.chromium.launch(args=["--no-sandbox"], **kw)
        pg = b.new_page()
        pg.expose_function("snap", lambda a: server._app_result(server._snapshot, a["path"], a.get("since", ""))
                           .model_dump(by_alias=True, exclude_none=True))  # page_snapshot's sync core
        pg.set_content(live_host(shown))
        app = pg.frame_locator("#app")
        view = app.frame_locator("#view")
        assert view.locator("#v").inner_text() == "one"
        assert view.locator("#js").inner_text() == "js ran"  # inlined script ran under the host CSP
        assert view.locator("#dot").evaluate("i => i.naturalWidth") == 1  # inlined image loaded
        assert "ui/notifications/size-changed" in pg.evaluate("window.got")
        time.sleep(0.05)
        server.write_page("wide", '<!doctype html><title>Wide</title><p id=v>two</p>', "html", create=False)
        view.locator("#v", has_text="two").wait_for(timeout=8000)  # the view reloaded by itself
        assert "live" in app.locator("#state").inner_text()
        server._delete("wide", True)
        app.locator("#state", has_text="stopped").wait_for(timeout=8000)  # a deleted page stops the polling

        # md pages keep the web UI's rule: raw <script> in the markdown never runs.
        pg = b.new_page()
        pg.set_content(live_host(md_first))
        frame = pg.frame_locator("#app").locator("#view[srcdoc]")
        frame.wait_for(state="attached")
        # No allow-scripts: the browser itself refuses the script (and Playwright can't enter the frame).
        assert frame.get_attribute("sandbox") == "allow-popups allow-popups-to-escape-sandbox"
        assert "id=m>md</p>" in frame.get_attribute("srcdoc").replace('"', "")
        b.close()
    print("OK: preview_page renders md and html pages, flags overflow; both MCP Apps work in a mock host,"
          " and the live view follows edits, stops on delete, keeps md scripts off; preview blocks local SSRF")
finally:
    server.stop()
