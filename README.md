# qillqa

A local notebook of nested pages with an MCP server, a web UI and local semantic search. *Qillqa* means "writing" in Quechua.

- **Page**: a folder in `pages/` (or `QILLQA_PAGES_DIR`) holding an `index.md` or `index.html`. Its subfolders are subpages, so pages nest to any depth. You can also edit the files by hand.
- **MCP** (`server.py`, stdio):
  - Writing: `create_page`, `update_page`, `attach_file`, `get_page`, `list_pages`, `move_page`, `delete_page`. Writes go straight to disk and need no server.
  - Searching: `search`. Pass an optional `path` to search only one folder (every folder can act as a project). Starts the server only if needed.
  - Server: `start_server`, `server_status`, `stop_server`.
  - Preview: `preview_page(path, width=1100)` returns a full-page PNG of how the page renders (390 = phone) plus its height and whether it scrolls sideways, so the agent can check a page before sharing it. Claude Code shows the image to the model. It is also an **MCP App**: hosts that render MCP Apps (Claude Desktop) show the screenshot inline with an "Open in qillqa" button (`ui://qillqa/preview`). Needs `playwright`; it uses Chrome at `QILLQA_CHROME` (default `~/.local/opt/google-chrome/chrome`) or Playwright's own Chromium.
- **Web UI** (`webui.py`): `http://127.0.0.1:47391`, with sign-in by the token in `.env`. Shows the page tree, the pages and search.
  - Next to the search box, **only in &lt;folder&gt;** limits the search to the folder you are viewing.
  - Every page renders sandboxed (CSP sandbox), so its JS cannot use your session.
  - **Files**: photos, videos, audio, PDFs, `.css`, `.js` and any other file go in the page folder (`attach_file`, or by hand). Reference them by relative path: `![photo](photo.jpg)`, `![demo](clip.mp4)` (video and audio get a player), `[report](report.pdf)`, `<link href="style.css">`.
  - **Links between pages**: `[text](#project/notes)` opens that page in place (a `#heading` of the same page stays an in-page anchor). Headings take a fixed anchor with `## Title {#my-id}`, and `<details markdown="1">` keeps markdown (code blocks, tables) inside a collapsible block.
  - Every page is served at a signed URL, `/f/<sig>/<folder>/`, which opens only that folder and its subpages. `/view?path=<path>` takes you there when you are signed in.
  - **Checklists**: in a markdown page, a `☐`, `☑` or `◐` in a table row, or at the start of a list item (`- ☐ …`), is a box you can click. A table row that holds a box also gets a ⠿ grip: drag it to another place or into another table (for example from "To do" to "Done"); the page scrolls by itself near the edges. Every change is saved to the page's `index.md` (`POST /api/page/edit`), which stays the source of truth. Each render carries its own capability (bound to that page and that version of the file): if the file changed since the page loaded, the change is refused and the page reloads, and no other document in the frame can edit. The MCP and the web UI write under one shared file lock, so neither loses the other's write. Line endings are kept. A box inside a sentence or a code block stays text. Give the tables of one page the same columns so a row fits wherever it lands. These pages run only a script carrying the page's nonce; a `<script>` written in the markdown never runs.
- **Semantic search** with `Qwen/Qwen3-Embedding-0.6B` (local, multilingual): on the GPU if there is one, otherwise on the CPU in fp32.
  - The index lives in `var/index-*.db`. It is updated by mtime at two moments: in the background after every MCP write, and before every search.
  - The model runs in a **child process** (`embedder.py`) that exits after `QILLQA_MODEL_IDLE_MIN` idle minutes, releasing all of its RAM and VRAM. The web server never loads torch.
  - Only results scoring at least `QILLQA_MIN_SCORE` are shown. The default of 0.3 is calibrated for this model: real hits score 0.41–0.65 and noise ≤ 0.21.

### Measured numbers (RTX 5060 Laptop, WSL)
| State | Time | Memory |
|---|---|---|
| Server up, no model | starts in 0.5–1 s | 76 MB RAM, 0 VRAM |
| Model active (after a write or search) | cold load ~6 s (~19 s the very first time on a machine) | +1.7 GB RAM, 1.36 GB VRAM |
| Search with the model loaded | ~30 ms | — |
| After the idle timeout | — | back to 77 MB, 0 VRAM |
| `stop_server` | 0.5 s | 0 |

## Install
```bash
git clone https://github.com/jordybc1503/qillqa && cd qillqa
bash setup.sh      # venv, torch (CUDA if there is an NVIDIA GPU, CPU otherwise), deps, .env with a token, model download
claude mcp add -s user qillqa -- "$PWD/.venv/bin/python" "$PWD/server.py"
```
Settings live in `.env` (see `.env.example`): host, port, token, pages folder, model, device, idle timeout and minimum score.
Changing the model creates a new index, because vectors from different models do not mix.

Your token: `grep QILLQA_TOKEN .env | cut -d= -f2-`

## Access from other devices (optional)
[Tailscale](https://tailscale.com) gives you a private HTTPS link, reachable only by your own devices, without opening ports:
```bash
sudo tailscale serve --bg 47391        # -> https://<machine>.<tailnet>.ts.net (still asks for the token)
sudo tailscale serve --https=443 off   # to remove it
```
The first time, Tailscale asks you to enable "Serve" on your tailnet. If you change the port, use your `QILLQA_PORT` value.

## Test
```bash
.venv/bin/python test_qillqa.py     # checks with a fake embedder, no model needed
.venv/bin/python test_preview.py    # preview_page screenshots + its MCP App in a mock host
python test_browser.py              # checklists in Chromium (any python with playwright; skips without it)
claude mcp list                     # qillqa should show "Connected"
```
