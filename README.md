# 🧠 PPT Agent — All Versions (v1 → v6)

A single README that documents every major iteration of your “PDF → summary → outline → PowerPoint” toolchain —
from the original CLI (v1) to the latest **LangGraph ReAct + Ollama** agent with a GUI runner (v6).

> This extends your previous “v1 → v5” overview with a new **v6** section.


---

## Overview

- **v1 — `ppt_builder.py` (LangChain + Ollama + python-pptx):** CLI that reads PDFs, summarizes, and renders slides locally.
- **v2 — “LangGraph PPT Agent” (`main.py`):** Lightweight agent loop; auto-detects Ollama models and builds a deck from a single prompt.
- **v3 — `ppt_agent.py` (single-file agent):** End-to-end research → notes → outline → PPT with stricter JSON outline and outline repair.
- **v4 — CPU‑First (Transformers stack):** Swaps to Hugging Face Transformers on CPU; keeps the LangGraph workflow and PPT renderer.
- **v5 — Kivy UI + MCP PowerPoint server:** Desktop app that launches the agent and builds decks via MCP tools; no `python-pptx` needed in the UI process.
- **v6 — LangGraph ReAct + Ollama + Kivy Runner (this repo):** Local-only agent with strict tools, **fallback build** logic, **verbose step tracing**, improved **Stop** behavior in the GUI runner, and compatibility fixes for `@tool` docstrings & outline/build typing.


---

## Quick Start (pick a version)

### v1 — `ppt_builder.py` (Ollama + python-pptx)
```bash
pip install langchain-core langchain-text-splitters langchain-ollama pydantic pypdf python-pptx requests
ollama serve && ollama pull llama3.1:8b
python ppt_builder.py --topic "AI Bias in Language Models" --pdf path/to/paper.pdf
```
Features: PDF parsing, chunking, LLM summarization, deck generation, and rendering with `python-pptx`.

### v2 — Agent loop (`main.py`)
```bash
pip install -r requirements.txt
ollama serve && ollama pull llama3.1:8b
python main.py
# e.g., type: make 11 slides ppt about dog
```
Auto-detects a local Ollama model and runs one interactive loop: list PDFs → summarize → build slides.

### v3 — `ppt_agent.py` (single‑file agent)
```bash
pip install langchain-ollama langgraph langchain-core pydantic pymupdf python-pptx
ollama serve && ollama pull llama3.1:8b
python ppt_agent.py
# try: summary only for "D:/papers/attention.pdf"
```
Adds robust slide‑outline JSON with title repair, deduping, and guaranteed Welcome → Content → Conclusion → Thanks structure.

### v4 — CPU‑First (HF Transformers)
```bash
pip install transformers torch langchain-huggingface langgraph langchain-core pydantic pymupdf python-pptx
export HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export HF_LOAD_4BIT=0
python ppt_agent.py
```
Runs on CPU by default; keeps read → summarize (map‑reduce) → outline (JSON) → build flow; good for laptops without CUDA.

### v5 — Kivy Desktop UI + MCP PowerPoint
```bash
pip install -U kivy langchain-ollama langgraph mcp PyMuPDF pydantic
python kivy_runner.py
# Browse to your ppt_agent.py, Start, then type: build slides now
```
Uses a local MCP PowerPoint server (via `uvx`) to render `.pptx` (templates auto‑picked or set via env).

### v6 — LangGraph ReAct + Ollama + Kivy Runner (local-only)
```bash
# Requirements
pip install "langchain>=0.3.0" "langgraph>=0.2.20" "langchain-community>=0.3.0" "langchain-ollama>=0.1.0" pypdf python-pptx
ollama pull gpt-oss:20b

# CLI
python ppt_agent.py --verbose
# Example flow:
# user > read pdf D:\work_on_github\langchain_pptBuilder\version5\nlp.pdf
# user > build ppt based on the paper

# GUI Runner
pip install kivy
python kivy_runner.py
```
**Highlights (v6):**
- **Strict tools** with docstrings (`@tool`) to satisfy LangChain v0.3+.
- **Type‑tolerant build**: `build_ppt` accepts **dict or JSON string** for outlines.
- **Pinned local model** for outlining (ignores accidental `gpt-4*` names; uses your Ollama model).
- **Fallback build**: if the model doesn’t call tools after “build ppt…”, the agent auto‑runs **outline → build** once using the cached summary/text.
- **Verbose trace** (`--verbose`): prints steps like `agent → tool` / `tool → agent` + short payloads.
- **Recursion guard** (`--recursion 24`) for deeper reasoning turns.
- **Durable Stop** (in Kivy runner): Windows uses **CTRL_BREAK_EVENT** + terminate/kill fallback; macOS/Linux uses **SIGINT → SIGTERM → SIGKILL** without closing stdin first (prevents EOFError tracebacks).
- **Cleaner REPL prompts**: `user >` for input and `agent>` for final replies; verbose streaming shows `agent message:` lines.

> Decks are written to `./decks/` and state is kept in `agent_state.json` (topic, last outline, last text/summary, last PPT path).


---

## Version‑by‑Version Details

### v1 — PPT Builder (CLI)
- **What it does:** Read academic PDFs, summarize with local Ollama, and render slides using `python-pptx`.
- **Install/Run:** Python 3.9+, Ollama running, and the listed packages; supports `--topic`, `--pdf`, `--slides`, etc.
- **How it works:** `pypdf` parse → chunk → summarize → reduce → deck schema → pptx.

### v2 — LangGraph PPT Agent (agent loop)
- **Highlights:** Auto‑detects Ollama; pluggable tools (list/summarize PDFs, knowledge fallback, create PPT).
- **Usage:** `python main.py` → natural language like “build slides now”; prints where the `.pptx` is saved.

### v3 — Single‑File Agent (`ppt_agent.py`)
- **Core:** Reads PDFs (page ranges), generates dense notes, creates strict slide‑outline JSON, and renders PPT.
- **Agent graph:** Planner chooses next action; tool node executes (read/summarize/outline/build/chat).

### v4 — CPU‑First Transformers
- **Shift:** Replace Ollama model calls with HF `pipeline("text-generation")` via LangChain’s `HuggingFacePipeline` / `ChatHuggingFace`.
- **CPU tips:** Use ≤7B models; keep 4‑bit off on CPU; `device_map` pinned to CPU.

### v5 — Kivy UI + MCP PowerPoint
- **UI flow:** Start `kivy_runner.py`, point to `ppt_agent.py`, send commands; output streams into the panel.
- **PPT rendering:** Calls an MCP server via `uvx`; saves timestamped `.pptx`.
- **Typical workflows:** Topic → slides, summary‑only, PDF → slides; supports “regenerate outline.”

### v6 — LangGraph ReAct + Ollama + Kivy Runner (local‑only)
- **What changed:** Stronger system policy to force tool use, strict JSON outlines, outline/build debouncing, and a **safe fallback** when the LLM stalls.
- **Runner:** Robust **Stop** (no stdin close, signals first), **Start** runs `ppt_agent.py --verbose`, and a clean Windows process-group setup for CTRL+BREAK.
- **Dev quality:** Docstrings for tools, sanitized filenames, outline caching, hash‑based debounce, and readable step logs.
- **Tips:** If the model spins, try `--recursion 24`; if outline repeats, increase `--slides` or provide a clearer topic/notes.

---

## Common Commands & Examples

- **Topic to slides (v2/v3/v4/v5/v6):** `make 11 slides ppt about dog`
- **Summary only (v3/v4/v6):** `summary only "D:/papers/attention.pdf"`
- **Page‑range read (v3/v4):** `read "D:/papers/paper.pdf" pages 1-3 and summarize`
- **Describe latest deck (v6):** `tell me about the ppt`

---

## Requirements Matrix (quick)

| Version | Runtime & Model | Key Deps |
|---|---|---|
| **v1** | Ollama local LLM | `langchain-*`, `pypdf`, `python-pptx` |
| **v2** | Ollama | `requirements.txt` (LangChain, python‑pptx if used) |
| **v3** | Ollama | `langchain-ollama`, `langgraph`, `pymupdf`, `python-pptx` |
| **v4** | HF Transformers (CPU) | `transformers`, `torch`, `langchain-huggingface`, `pymupdf`, `python-pptx` |
| **v5** | Ollama (agent) + MCP server (PPT) | `kivy`, `langchain-ollama`, `langgraph`, `mcp`, `PyMuPDF` |
| **v6** | Ollama (agent) + python‑pptx + Kivy runner | `langchain>=0.3`, `langgraph>=0.2.20`, `langchain-ollama`, `pypdf`, `python-pptx`, `kivy` |

---

## Troubleshooting (highlights)

- **No models / can’t connect (Ollama)** → ensure `ollama serve` and pull at least one model (`gpt-oss:20b` by default).
- **Too few agent steps** → run with `--recursion 24` (or 30) for deeper chains.
- **Outline tool tries gpt-4** → v6 pins the outline tool to your **local Ollama** model.
- **`@tool` docstring error** → every tool has a docstring in v6; if you add tools, include one.
- **Validation error: build_ppt expects a string** → v6 build accepts **dict or string**.
- **EOF after Stop** → v6 Kivy runner signals first (no stdin close); add an `EOFError` guard to the CLI loop if needed.

---

## Layouts (by version)

- **v1:** `ppt_builder.py` (+ README)
- **v2:** `main.py` (+ optional tools)
- **v3/v4:** `ppt_agent.py` (tools + graph + CLI)
- **v5:** `kivy_runner.py` + `ppt_agent.py` (MCP render path)
- **v6:** `ppt_agent.py` + `kivy_runner.py` (local render with `python-pptx`, robust Stop, verbose steps)

---

## License

MIT (or your preferred license). Update this section to match your repository’s license.
