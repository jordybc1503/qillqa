# qillqa

Cuaderno local de páginas anidadas con MCP, web y búsqueda semántica. *Qillqa* significa "escritura" en quechua.

- **Página** = carpeta en `pages/` (o en `QILLQA_PAGES_DIR`) con `index.md` o `index.html`. Sus subcarpetas son subpáginas. También puedes editar los archivos a mano.
- **MCP** (`server.py`, stdio):
  - Para escribir: `create_page`, `update_page`, `attach_file`, `get_page`, `list_pages`, `move_page`, `delete_page`. Escribe directo en disco; no necesita el servidor.
  - Para buscar: `search`. Arranca el servidor solo si hace falta.
  - Para el servidor: `start_server`, `server_status`, `stop_server`.
- **Web** (`webui.py`): `http://127.0.0.1:47391`, con login por el token de `.env`. Muestra el árbol, las páginas y la búsqueda.
  - Cada página se muestra aislada (CSP sandbox), así que su JS no puede usar tu sesión.
  - **Archivos**: fotos, vídeos, audio, PDFs, `.css`, `.js` o cualquier otro archivo van en la carpeta de la página (`attach_file` o a mano). Se referencian con ruta relativa: `![foto](foto.jpg)`, `![demo](clip.mp4)` (vídeo y audio salen con reproductor), `[informe](informe.pdf)`, `<link href="estilo.css">`.
  - Cada página se sirve en una URL firmada, `/f/<firma>/<carpeta>/`, que solo abre esa carpeta y sus subpáginas. `/view?path=<ruta>` te lleva a ella si tienes sesión.
- **Búsqueda semántica** con `Qwen/Qwen3-Embedding-0.6B` (local, multilingüe), en la GPU si hay; si no, en CPU con fp32.
  - El índice vive en `var/index-*.db`. Se actualiza por mtime en dos momentos: en segundo plano tras cada escritura del MCP y antes de cada búsqueda.
  - El modelo corre en un **proceso hijo** (`embedder.py`) que se cierra tras `QILLQA_MODEL_IDLE_MIN` minutos sin uso, y entonces se libera toda su RAM y VRAM. El servidor web nunca carga torch.
  - Solo muestra resultados con score ≥ `QILLQA_MIN_SCORE` (0,3, calibrado para este modelo: los aciertos dan 0,41–0,65 y el ruido ≤ 0,21).

### Números medidos (RTX 5060 Laptop, WSL)
| Estado | Tiempo | Memoria |
|---|---|---|
| Servidor arriba, sin modelo | arranca en 0,5–1 s | 76 MB de RAM, 0 de VRAM |
| Modelo activo (tras crear o buscar) | carga en frío ~6 s (~19 s la primera vez en la máquina) | +1,7 GB de RAM, 1,36 GB de VRAM |
| Búsqueda con el modelo cargado | ~30 ms | — |
| Tras la inactividad | — | vuelve a 77 MB, 0 de VRAM |
| `stop_server` | 0,5 s | 0 |

## Instalar
```bash
bash setup.sh                       # venv + torch CPU + deps, .env con token, descarga del modelo
claude mcp add -s user qillqa -- ~/code/mcp/qillqa/.venv/bin/python ~/code/mcp/qillqa/server.py
```
Configuración en `.env` (ver `.env.example`): host, puerto, token, carpeta de páginas, modelo e inactividad.
Si cambias de modelo se crea un índice nuevo, porque los vectores de modelos distintos no se mezclan.

## Acceso desde otros equipos (opcional)
Con [Tailscale](https://tailscale.com) obtienes un link HTTPS privado, solo para tus dispositivos, sin abrir puertos:
```bash
sudo tailscale serve --bg 47391        # → https://<maquina>.<red>.ts.net (pide el token igual)
sudo tailscale serve --https=443 off   # para quitarlo
```
La primera vez, Tailscale pide activar "Serve" en tu red. Si cambias de puerto, usa el valor de `QILLQA_PORT`.

## Probar
```bash
.venv/bin/python test_qillqa.py     # checks sin modelo (embedder falso)
python3 ../_smoke_test.py qillqa    # el MCP arranca y lista sus tools
```
