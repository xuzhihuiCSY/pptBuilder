# Version 5

A tiny Kivy desktop app (`kivy_runner.py`) that launches and talks to your `ppt_agent.py` process over stdin/stdout. The agent reads/summarizes PDFs, generates a slide outline, and builds a PowerPoint deck via an MCP PowerPoint server—all locally.

---

## Project Layout

```
.
├── kivy_runner.py   # Kivy UI: start/stop agent, send commands, view output
└── ppt_agent.py     # The agent: read/summarize/outline/build PPT via tools
```

## What You Can Do

- **Run the agent with a GUI**—no terminal required.
- **Read PDFs** and extract text.
- **Summarize** long content (exec prose or slides-focused).
- **Generate slide outlines** (balanced use of bullets/quote/section).
- **Render .pptx decks** using an **MCP PowerPoint** server (no `python-pptx` needed).

> The agent uses LangChain + Ollama locally and talks to a PowerPoint MCP server (invoked via `uvx`).

---

## Quick Start (Windows/macOS/Linux)

1) **Create & activate a virtual environment** (recommended):

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
```

2) **Install dependencies**

> If you use `uv` you can keep things isolated and fast, but plain `pip` is fine.

```bash
pip install -U kivy langchain-ollama langgraph mcp PyMuPDF pydantic
# Optional (if you plan to run the MCP PowerPoint server via uvx):
pip install -U uv
```

3) **Run the Kivy UI**

```bash
python kivy_runner.py
```

- Click **Browse** and select your local `ppt_agent.py`.
- Click **Start** to launch the agent. Output will stream in the bottom panel.
- Type a command (e.g., `build slides now` or `summary only "C:\\path\\to\\paper.pdf"`) and press **Enter** or **Send**.

4) **PowerPoint building**

The agent calls an **MCP PowerPoint server** using `uvx` (zero‑install style):

- It will try: `uvx --from office-powerpoint-mcp-server ppt_mcp_server`
- Ensure `uv` is installed (`pip install uv`) and your environment can run `uvx`.
- Templates are auto‑picked (`general`, `relax`, `research`) or you can set one via env (see below).

> Output `.pptx` files are saved next to your working directory with a timestamped name.

---

## Typical Workflows

### A) Slides from a topic (no PDF)
1. Start the agent from the Kivy UI.
2. Send: `Build slides now about "Diffusion Models in Practice" (11 slides)`  
   The agent will synthesize notes → outline → build a deck automatically.

### B) Summary only (from PDF)
1. Send: `summary only "C:\\Users\\me\\Desktop\\paper.pdf"`  
2. The agent reads the PDF and returns an executive summary (no slides).

### C) PDF → Slides (guided)
1. Send something like: `Make a 12‑slide deck from "D:\\notes\\report.pdf"`  
2. The agent will read → summarize → outline → (auto) build the `.pptx`.

> You can also ask it to **regenerate the outline** if you want a fresh structure.

---

## Environment Variables (tuning)

You can set these before starting the agent to influence behavior:

- `OLLAMA_MODEL` – e.g., `gpt-oss:20b` (default)  
- `OLLAMA_BASE_URL` – e.g., `http://127.0.0.1:11434`
- `NUMERIC_STYLE` – `strict` | `balanced` (default) | `light`
- `SUMMARY_SOURCE_CAP` – cap source chars (0 = unlimited)
- `SUMMARY_CHUNK_SIZE` – map chunk size (default ~3000 chars)
- `SUMMARY_MAX_CHUNKS` – limit number of chunks (0 = unlimited)
- `KEEP_OUTLINE_AFTER_BUILD` – set to `1` to keep outline after building
- `PPT_TEMPLATE_NAME` – template alias or filename (e.g., `general`, `relax`, `research`, or a path like `templates/template_general.pptx`)

Windows PowerShell example:

```powershell
$env:OLLAMA_MODEL="gpt-oss:20b"
$env:NUMERIC_STYLE="balanced"
python kivy_runner.py
```

---

## Dependencies & Notes

- **Kivy** for the desktop UI (`kivy_runner.py`).
- **LangChain / LangGraph** orchestrate LLM calls and tool flow.
- **Ollama** provides local LLM inference—make sure your Ollama daemon is running if you use it.
- **PyMuPDF (`fitz`)** for reading PDF text.
- **MCP** client for tool calls to a PowerPoint server process.
- **uvx** for running `office-powerpoint-mcp-server` without a global install.

> If you prefer, you can install the MCP PowerPoint server locally and run `ppt_mcp_server` directly; the agent will try to detect tool names such as `create_presentation_from_template`, `add_slide`, `populate_placeholder`, and `save_presentation`.

---

## Troubleshooting

- **Nothing shows in the terminal panel**  
  - Ensure the agent path is correct and that `ppt_agent.py` runs with your current Python.
  - The Kivy app forces UTF‑8 (`PYTHONUTF8=1`) and streams stdout/stderr into the UI.

- **“MCP PowerPoint tools not found”**  
  - Install `uv` and try again: `pip install uv` (or use your package manager).  
  - Confirm `uvx --from office-powerpoint-mcp-server ppt_mcp_server` runs from a terminal.

- **PDF not found / access denied**  
  - Use an absolute path in quotes. On Windows: `"C:\\Users\\me\\file.pdf"`.

- **Weird titles like “Slide 3”**  
  - The agent repairs generic titles and tries to inject better ones from keyphrases. You can also ask it to “regenerate outline”.

- **Too many numbers or too few**  
  - Adjust `NUMERIC_STYLE` (try `strict` or `light`).

---

## Security & Privacy

- Everything runs **locally** unless your Ollama or MCP endpoints route elsewhere.  
- PDFs are read directly on disk; nothing is uploaded by default.

---

## License

Add your preferred license here (MIT/Apache-2.0/etc.).
