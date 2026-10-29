"""preview_page and its MCP App: .venv/bin/python test_preview.py
The screenshot part needs playwright (and Chrome/Chromium); it skips without it."""
import asyncio
import json
import os
import socket
import struct
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
(tmp / "pages").mkdir()
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
PORT = sock.getsockname()[1]
sock.close()
os.environ.update(QILLQA_PAGES_DIR=str(tmp / "pages"), QILLQA_TOKEN="t0k-preview", QILLQA_PORT=str(PORT))

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
    </script>""".replace("__UI__", json.dumps(html.content).replace("</", "<\\/")).replace("__RESULT__", json.dumps(result))
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
    print("OK: preview_page renders md and html pages, flags overflow, and its MCP App works in a mock host")
finally:
    server.stop()
