# 🧠 PPT Agent — All Versions (v1 → v5)

A single README that documents every major iteration of your “PDF → summary → outline → PowerPoint” toolchain—from the original CLI (v1) to the Kivy desktop UI (v5).

---

## Overview

- **v1 — `ppt_builder.py` (LangChain + Ollama + python-pptx):** CLI that reads PDFs, summarizes, and renders slides locally.
- **v2 — “LangGraph PPT Agent” (`main.py`):** Lightweight agent loop; auto-detects Ollama models and builds a deck from a single prompt.
- **v3 — `ppt_agent.py` (single-file agent):** End-to-end research → notes → outline → PPT with stricter JSON outline and outline repair.
- **v4 — CPU‑First (Transformers stack):** Swaps to Hugging Face Transformers on CPU; keeps the LangGraph workflow and PPT renderer.
- **v5 — Kivy UI + MCP PowerPoint server:** Desktop app that launches the agent and builds decks via MCP tools; no `python-pptx` needed in the UI process.

---

## Quick Start (pick one version)

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

---

## Common Commands & Examples

- **Topic to slides (v2/v3/v4/v5):** `make 11 slides ppt about dog`
- **Summary only (v3/v4/v5):** `summary only "D:/papers/attention.pdf"`
- **Page‑range read (v3/v4):** `read "D:/papers/paper.pdf" pages 1-3 and summarize`

---

## Requirements Matrix (quick)

| Version | Runtime & Model | Key Deps |
|---|---|---|
| **v1** | Ollama local LLM | `langchain-*`, `pypdf`, `python-pptx` |
| **v2** | Ollama | `requirements.txt` (LangChain, python‑pptx if used) |
| **v3** | Ollama | `langchain-ollama`, `langgraph`, `pymupdf`, `python-pptx` |
| **v4** | HF Transformers (CPU) | `transformers`, `torch`, `langchain-huggingface`, `pymupdf`, `python-pptx` |
| **v5** | Ollama (agent) + MCP server (PPT) | `kivy`, `langchain-ollama`, `langgraph`, `mcp`, `PyMuPDF`; optional `uv` |

---

## Roadmap & Quality Controls

- Exact slide counts, cleaned titles, deduped bullets, numeric‑fact injection, and recoveries if the LLM returns bad JSON (v3+).
- Roadmap: CPU tuning, small desktop/web UI, templates, and multi‑doc synthesis.

---

## Troubleshooting (highlights)

- **No models / can’t connect (Ollama)** → ensure `ollama serve` and pull at least one model.
- **Slow on CPU (v4)** → prefer 7B, adjust chunk sizes.
- **MCP PowerPoint not found (v5)** → install `uv`, ensure `uvx` runs the MCP server.

---

## Project Layouts

- **v1:** `ppt_builder.py` (+ README)
- **v2:** `main.py` (+ optional tools)
- **v3/v4:** `ppt_agent.py` (tools + graph + CLI)
- **v5:** `kivy_runner.py` + `ppt_agent.py`

---

## License

MIT (or your preferred license). Update this section to match your repository’s license.
