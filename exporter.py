"""Export a page — or a page and its subpages, or a folder — to ONE self-contained .html file.

  .venv/bin/python exporter.py <path> [--subpages] [-o out.html]

The file opens anywhere (double click, WhatsApp, e-mail) without qillqa or a server:
- every file of every page is embedded (base64, one <script type="text/plain"> each, decoded
  only when its page opens);
- each page runs in a srcdoc iframe under a virtual base https://qillqa.invalid/<page>/, and a
  small shim maps every request for one of its files (fetch, XHR, img/script/link src, innerHTML)
  to a blob: URL;
- ES modules from a CDN (import maps, <script src=https://…>) are fetched at export time and
  embedded too, so 3D pages work offline. If a fetch fails the online URL stays and the report
  says the file needs internet.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import json
import mimetypes
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from server import INDEX_NAMES, PAGES, VAR, _index, _rel, _safe, _title

BASE = "https://qillqa.invalid/"
WARN_BYTES = 40 * 2**20
CDN_HOSTS = {"cdn.jsdelivr.net", "unpkg.com", "esm.sh", "cdnjs.cloudflare.com", "ga.jspm.io", "cdn.skypack.dev"}
for ext, mime in ((".glb", "model/gltf-binary"), (".gltf", "model/gltf+json"), (".js", "text/javascript"),
                  (".mjs", "text/javascript"), (".wasm", "application/wasm"), (".webp", "image/webp"),
                  (".hdr", "image/vnd.radiance"), (".md", "text/markdown")):
    mimetypes.add_type(mime, ext)

# import/export specifiers in JS: from '…', import '…', import('…')
_IMPORT = re.compile(r"""(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"\n]+)\2""")
_MODULE = re.compile(r"""<script\b(?=[^>]*\btype\s*=\s*["']?module)[^>]*>(.*?)</script>""", re.I | re.S)
_IMPORTMAP = re.compile(r"""<script\b[^>]*\btype\s*=\s*["']?importmap["']?[^>]*>(.*?)</script>""", re.I | re.S)
_SCRIPT_SRC = re.compile(r"""<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.I)


def _pages(d: Path, subpages: bool) -> list[Path]:
    """The page itself (if it has an index) plus, with subpages, every descendant page."""
    out = [d] if _index(d) else []
    if subpages or not out:  # a folder with no page of its own exports what is under it
        out += sorted(p for p in d.rglob("*") if p.is_dir() and not any(x.startswith(".") for x in p.relative_to(d).parts)
                      and _index(p))
    return out


def _files(page: Path) -> list[Path]:
    """The page's own files: everything under its folder except its index and its subpages."""
    out = []
    for f in sorted(page.rglob("*")):
        rel = f.relative_to(page)
        if not f.is_file() or any(x.startswith(".") for x in rel.parts) or (len(rel.parts) == 1 and f.name in INDEX_NAMES):
            continue
        if any(_index(page.joinpath(*rel.parts[:n])) for n in range(1, len(rel.parts))):
            continue  # inside a subpage: it belongs to that page
        out.append(f)
    return out


def _page_html(page: Path) -> str:
    idx = _index(page)
    text = idx.read_bytes().decode("utf-8", "replace")
    if idx.suffix == ".html":
        return text
    import webui  # lazy: only the markdown renderer is needed
    return webui._render_md(_title(page), text)


def _allowed(url: str) -> bool:
    u = urllib.parse.urlparse(url)
    return u.scheme == "https" and (u.hostname or "") in CDN_HOSTS


def _fetch(url: str) -> bytes | None:
    """A CDN file, cached under var/export-cache. None if it is not on an allowed public CDN
    (a page must never make the server fetch 127.0.0.1 or the LAN into a shared file) or
    cannot be fetched."""
    if not _allowed(url):
        return None
    cache = VAR / "export-cache" / hashlib.sha1(url.encode()).hexdigest()
    if cache.exists():
        return cache.read_bytes()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qillqa-export"}), timeout=20) as r:
            if not _allowed(r.geturl()):  # a redirect left the allowlist
                return None
            data = r.read()
    except Exception:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


def _resolve(spec: str, base: str, imap: dict) -> str | None:
    """Specifier -> absolute URL, like the browser: relative to base, bare through the import map."""
    if spec.startswith(("./", "../", "/")) or re.match(r"^[a-z][\w+.-]*:", spec, re.I):
        return urllib.parse.urljoin(base, spec)
    if spec in imap:
        return urllib.parse.urljoin(base, imap[spec])
    pre = max((k for k in imap if k.endswith("/") and spec.startswith(k)), key=len, default=None)
    return urllib.parse.urljoin(base, imap[pre] + spec[len(pre):]) if pre else None


def _rewrite_js(src: str, url: str) -> str:
    """A module will run from a blob: URL, so its relative specifiers and import.meta.url must
    be made absolute first (the import map then sends those absolute URLs to blobs)."""
    def fix(m):
        spec = m.group(3)
        if spec.startswith(("./", "../", "/")):
            return f"{m.group(1)}{m.group(2)}{urllib.parse.urljoin(url, spec)}{m.group(2)}"
        return m.group(0)
    return _IMPORT.sub(fix, src).replace("import.meta.url", json.dumps(url))


class _Bundle:
    def __init__(self):
        self.blobs: list[tuple[str, bytes]] = []  # (mime, bytes); index = id
        self.ids: dict[str, int] = {}  # sha1 -> id (same bytes stored once)
        self.cdn: dict[str, int] = {}  # absolute CDN URL -> id
        self.missing: list[str] = []

    def add(self, data: bytes, mime: str) -> int:
        h = hashlib.sha1(data).hexdigest()
        if h not in self.ids:
            self.ids[h] = len(self.blobs)
            self.blobs.append((mime, data))
        return self.ids[h]

    def module(self, url: str, imap: dict, imports: dict, seen: set) -> None:
        """Fetch a CDN module and, recursively, what it imports."""
        if url in seen or len(seen) > 400:
            return
        seen.add(url)
        data = _fetch(url)
        if data is None:
            self.missing.append(url)
            return
        src = data.decode("utf-8", "replace")
        self._scan(src, url, imap, imports, seen)
        self.cdn[url] = self.add(_rewrite_js(src, url).encode(), "text/javascript")

    def _scan(self, src: str, base: str, imap: dict, imports: dict, seen: set) -> None:
        for m in _IMPORT.finditer(src):
            spec = m.group(3)
            url = _resolve(spec, base, imap)
            if not url:
                continue
            if not spec.startswith(("./", "../", "/")) and not re.match(r"^[a-z][\w+.-]*:", spec, re.I):
                imports[spec] = url  # bare: the import map needs this exact key (maps don't chain)
            ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if url.startswith(("http://", "https://")) and not url.startswith(BASE) and ext in ("", ".js", ".mjs"):
                self.module(url, imap, imports, seen)  # other extensions: example URLs in comments


def build(path: str, subpages: bool = False) -> tuple[str, bytes, dict]:
    """(file name, html bytes, report)."""
    root = _safe(path)
    if root == PAGES:
        raise ValueError("pick a page or a folder, not the whole notebook")
    pages = _pages(root, subpages)
    if not pages:
        raise ValueError(f"no page at {path!r}")
    b, out_pages, sizes = _Bundle(), [], []
    for page in pages:
        rel = _rel(page)
        base = BASE + urllib.parse.quote(rel) + "/"
        page_html = _page_html(page)
        maps = [json.loads(m) for m in _IMPORTMAP.findall(page_html) if m.strip()]
        imap = {k: v for m in maps for k, v in m.get("imports", {}).items()}
        imports: dict[str, str] = {}
        files: dict[str, int] = {}
        seen: set[str] = set()
        for f in _files(page):
            url = base + urllib.parse.quote(f.relative_to(page).as_posix())
            data = f.read_bytes()
            mime = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if mime == "text/javascript":
                src = data.decode("utf-8", "replace")
                b._scan(src, url, imap, imports, seen)
                data = _rewrite_js(src, url).encode()
            files[url] = b.add(data, mime)
            sizes.append((f"{rel}/{f.relative_to(page).as_posix()}", len(data)))
        for body in _MODULE.findall(page_html):
            b._scan(body, base, imap, imports, seen)
        for src in _SCRIPT_SRC.findall(page_html):  # classic CDN scripts
            url = urllib.parse.urljoin(base, src)
            if url.startswith(("http://", "https://")) and not url.startswith(BASE) and url not in b.cdn:
                data = _fetch(url)
                if data is None:
                    b.missing.append(url)
                else:
                    b.cdn[url] = b.add(data, "text/javascript")
        for spec, target in imap.items():  # keep online fallbacks for what was not resolved
            imports.setdefault(spec, urllib.parse.urljoin(base, target))
        out_pages.append({"path": rel, "title": _title(page), "base": base, "md": _index(page).suffix == ".md",
                          "html": b.add(page_html.encode(), "text/html"), "files": files, "imports": imports,
                          "depth": len(page.relative_to(root).parts)})
    name = (root.name or "qillqa") + ".html"
    doc = _document(_title(root) if _index(root) else root.name, out_pages, b)
    total = len(doc)
    report = {"file": name, "bytes": total, "pages": [p["path"] for p in out_pages],
              "offline": not b.missing, "needs_internet_for": sorted(set(b.missing)),
              "largest": [f"{n} ({s / 2**20:.1f} MB)" for n, s in sorted(sizes, key=lambda x: -x[1])[:5]],
              "warning": f"{total / 2**20:.0f} MB: big for WhatsApp/e-mail" if total > WARN_BYTES else ""}
    return name, doc, report


def _document(title: str, pages: list[dict], b: _Bundle) -> bytes:
    manifest = {"pages": pages, "cdn": b.cdn, "mime": [m for m, _ in b.blobs],
                "exported": dt.date.today().isoformat()}
    blobs = "".join(f'<script type="text/plain" id="qf{i}">{base64.b64encode(d).decode()}</script>\n'
                    for i, (_, d) in enumerate(b.blobs))
    data = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    return (f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body class="{'multi' if len(pages) > 1 else 'single'}">
<aside id="nav"><div class="brand">{html.escape(title)}<small>qillqa · {manifest['exported']}</small></div><div id="list"></div></aside>
<button id="menu" type="button" aria-label="Pages">☰</button>
<main><div id="wait">Loading…</div><iframe id="view" title="{html.escape(title)}" allow="fullscreen"></iframe></main>
<script type="application/json" id="qx">{data}</script>
{blobs}<script>{_LOADER}</script></body></html>""").encode()


_CSS = """*{box-sizing:border-box}html,body{margin:0;height:100%;font:15px/1.5 system-ui,sans-serif;background:#f4f6fb;color:#1d2433}
main{position:fixed;inset:0;background:#fff}body.multi main{left:260px}iframe{border:0;width:100%;height:100%;display:block}
#wait{position:absolute;inset:0;display:grid;place-content:center;color:#5d6680}#wait[hidden]{display:none}
#nav{position:fixed;top:0;bottom:0;left:0;width:260px;overflow:auto;background:#fbfcff;border-right:1px solid #e2e6f0;padding:14px 10px}
body.single #nav,body.single #menu{display:none}
.brand{font-weight:700;padding:4px 8px 12px;color:#5b3fd0}.brand small{display:block;font-weight:400;color:#5d6680;font-size:.8em}
#list a{display:block;padding:6px 8px;border-radius:8px;color:#1d2433;text-decoration:none;overflow-wrap:anywhere}
#list a:hover{background:#eef0fa}#list a.on{background:#5b3fd0;color:#fff}
#menu{display:none;position:fixed;right:14px;bottom:14px;z-index:3;width:48px;height:48px;border-radius:50%;border:0;
background:#5b3fd0;color:#fff;font-size:20px;box-shadow:0 4px 14px #0004}
@media (max-width:760px){body.multi main{left:0}body.multi #menu{display:block}#nav{z-index:2;transform:translateX(-100%);transition:transform .2s}
body.open #nav{transform:none;box-shadow:0 0 40px #0005}}"""

# Runs in the exported file. Blob URLs are made only for the page being opened.
_LOADER = r"""(()=>{
const D=JSON.parse(document.getElementById('qx').textContent),$=id=>document.getElementById(id),made={};
const data=id=>'data:'+D.mime[id]+';base64,'+$('qf'+id).textContent;
async function url(id){if(!made[id])made[id]=fetch(data(id)).then(r=>r.blob()).then(b=>URL.createObjectURL(new Blob([b],{type:D.mime[id]})));return made[id]}
const text=id=>fetch(data(id)).then(r=>r.text());
const SHIM=String.raw`(()=>{const M=window.__qx;
const r=u=>{try{if(u==null)return null;const x=new URL(String(u),document.baseURI);x.hash='';x.search='';return M[x.href]||null}catch(e){return null}};
const F=window.fetch;window.fetch=function(i,o){const b=r(i instanceof Request?i.url:i);return b?F.call(this,b,i instanceof Request?{method:'GET'}:o):F.apply(this,arguments)};
const O=XMLHttpRequest.prototype.open;XMLHttpRequest.prototype.open=function(m,u,...a){return O.call(this,m,r(u)||u,...a)};
for(const[C,p]of[['HTMLImageElement','src'],['HTMLMediaElement','src'],['HTMLSourceElement','src'],['HTMLScriptElement','src'],['HTMLLinkElement','href'],
  ['HTMLAnchorElement','href'],['HTMLIFrameElement','src'],['HTMLVideoElement','poster'],['HTMLEmbedElement','src'],['HTMLObjectElement','data']]){
  const P=window[C]&&window[C].prototype,d=P&&Object.getOwnPropertyDescriptor(P,p);if(!d||!d.set)continue;
  Object.defineProperty(P,p,{...d,set(v){d.set.call(this,r(v)||v)}})}
const S=Element.prototype.setAttribute;Element.prototype.setAttribute=function(n,v){return S.call(this,n,/^(src|href|poster|data)$/i.test(n)?(r(v)||v):v)};
const fix=s=>typeof s!=='string'?s:s.replace(/(\s(?:src|href|poster)\s*=\s*)(["'])([^"']*)\2/gi,(m,a,q,v)=>{const b=r(v);return b?a+q+b+q:m})
  .replace(/url\(\s*(["']?)([^"')]+)\1\s*\)/g,(m,q,v)=>{const b=r(v);return b?'url('+q+b+q+')':m});
for(const p of['innerHTML','outerHTML']){const d=Object.getOwnPropertyDescriptor(Element.prototype,p);
  Object.defineProperty(Element.prototype,p,{...d,set(v){d.set.call(this,fix(v))}})}
const I=Element.prototype.insertAdjacentHTML;Element.prototype.insertAdjacentHTML=function(w,s){return I.call(this,w,fix(s))};
document.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('a[href]');if(!a)return;
  const h=a.getAttribute('href')||'',m=h.match(/^\/?#(.+)$/);
  if(m&&a.target==='_top'){e.preventDefault();parent.postMessage({qx:decodeURIComponent(m[1])},'*')}
  else if(h[0]==='#'&&h.length>1){e.preventDefault();const t=document.getElementById(decodeURIComponent(h.slice(1)));t&&t.scrollIntoView()}},true);  // the <base> would send #x elsewhere
})();`;
const byPath=Object.fromEntries(D.pages.map(p=>[p.path,p]));
async function open(path){
  const p=byPath[path]||D.pages[0];$('wait').hidden=false;
  document.querySelectorAll('#list a').forEach(a=>a.classList.toggle('on',a.dataset.p===p.path));
  const map={},ids=Object.entries(p.files).concat(Object.entries(D.cdn));
  await Promise.all(ids.map(async([u,id])=>{map[u]=await url(id)}));
  const imports={};for(const[s,u]of Object.entries(p.imports))imports[s]=map[u]||u;
  for(const u in map)if(/\.m?js$/i.test(u)||u in D.cdn)imports[u]=map[u];
  let h=(await text(p.html)).replace(/<script\b[^>]*\btype\s*=\s*["']?importmap["']?[^>]*>[\s\S]*?<\/script>/gi,'');
  const res=v=>{try{const x=new URL(v,p.base);x.hash='';x.search='';return map[x.href]}catch(e){}};
  const here=location.href.split('#')[0];
  h=h.replace(/(\s(?:src|href|poster|data)\s*=\s*)(["'])([^"']*)\2/gi,(m,a,q,v)=>{
       const pl=/href/i.test(a)&&v.match(/^\/?#(.+)$/);  // a link to another notebook page
       if(pl){const t=decodeURIComponent(pl[1]);return byPath[t]?a+q+here+'#'+encodeURIComponent(t)+q:a.replace(/href/i,'data-href')+q+v+q}
       const b=res(v);return b?a+q+b+q:m})
     .replace(/url\(\s*(["']?)([^"')]+)\1\s*\)/g,(m,q,v)=>{const b=res(v);return b?'url('+q+b+q+')':m});
  // markdown pages: no scripts at all (as in qillqa), so no shim; their links were rewritten above
  const head=p.md?'':'<base href="'+p.base+'"><script>window.__qx='+JSON.stringify(map).replace(/</g,'\\u003c')+';'+SHIM+'<\/script>'+
    '<script type="importmap">'+JSON.stringify({imports}).replace(/</g,'\\u003c')+'<\/script>';
  h=/<head[^>]*>/i.test(h)?h.replace(/<head[^>]*>/i,m=>m+head):head+h;
  const f=$('view');f.onload=()=>{$('wait').hidden=true};
  if(p.md)f.setAttribute('sandbox','allow-same-origin allow-popups allow-popups-to-escape-sandbox allow-top-navigation-by-user-activation');
  else f.removeAttribute('sandbox');
  f.srcdoc=h;document.title=p.title;
  document.body.classList.remove('open');if(location.hash.slice(1)!==encodeURIComponent(p.path))history.replaceState(null,'','#'+encodeURIComponent(p.path));
}
for(const p of D.pages){const a=document.createElement('a');a.href='#'+encodeURIComponent(p.path);a.dataset.p=p.path;
  a.textContent=p.title;a.style.paddingLeft=(8+14*p.depth)+'px';a.onclick=e=>{e.preventDefault();open(p.path)};$('list').append(a)}
$('menu').onclick=()=>document.body.classList.toggle('open');
addEventListener('message',e=>{if(e.data&&e.data.qx&&byPath[e.data.qx])open(e.data.qx)});
addEventListener('hashchange',()=>open(decodeURIComponent(location.hash.slice(1))));  // markdown page links
open(decodeURIComponent(location.hash.slice(1)));
})();"""


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("path")
    a.add_argument("--subpages", action="store_true", help="include every subpage")
    a.add_argument("-o", "--out", help="output file (default: var/exports/<name>.html)")
    o = a.parse_args()
    name, doc, report = build(o.path, o.subpages)
    out = Path(o.out) if o.out else VAR / "exports" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(doc)
    print(json.dumps({**report, "out": str(out)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
