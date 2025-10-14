# PPT Builder / LangGraph PPT Agent — Multi‑Version Guide

This repository contains **three related command‑line tools** for turning PDFs or topics into PowerPoint decks. Each version lives in its own folder and targets a different scope and maturity level.

```
.
├─ version1/   # Classic PPT Builder (non‑agent)
├─ version2/   # Agent v2 (lightweight tool-calling)
└─ version3/   # Agent v3 (LangGraph state machine)
```

> TL;DR: **v1** is stable and will not change. **v2** and **v3** are the active lines for improvements—especially PDF reading accuracy & speed and PPT beautification. **v3** will also grow new features like “summary‑only” and other power‑user flows.

---

## 🧭 Which version should I use?

| Need | Use this |
|---|---|
| Simple “topic → deck” or “PDF → deck” with minimal moving parts | **version1** |
| Agentic workflow with tool calls, local LLMs, and a conversational CLI | **version2** |
| A **LangGraph** state machine that plans actions (read → summarize → outline → build), fixes bad slides, and writes strong JSON outlines | **version3** |

---

## 📦 Common Requirements (all versions)

- **Python 3.9+**
- **Ollama** running locally (`http://localhost:11434`) with at least one pulled model (e.g., `llama3.1:8b`):
  ```bash
  ollama serve
  ollama pull llama3.1:8b
  ```
- **python-pptx** for rendering decks

Create a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## 📁 version1 — PPT Builder (non‑agent, stable)

A straightforward pipeline that reads a PDF (or a topic), summarizes, and renders a professional PowerPoint deck. It’s **not** an agent and is **frozen** (no planned changes).

**Highlights (from v1 README):**
- Reads academic PDFs (no OCR needed), summarizes with your local Ollama model, and builds slides via `python-pptx`.
- CLI flags for topic, audience, tone, slide count, model, template, and output filename.
- Optionally adds a References slide.  
Source: fileciteturn2file2

**Install deps (example):**
```bash
pip install langchain-core langchain-text-splitters langchain-ollama pydantic pypdf python-pptx requests
```

**Run examples:**
```bash
# Topic only
python ppt_builder.py --topic "AI Bias in Language Models"

# From a specific paper
python ppt_builder.py --topic "Inversion Models" --pdf path/to/paper.pdf
```

> ✅ **Status:** Stable; **no further changes planned** (kept for compatibility and simplicity).

---

## 🤖 version2 — Agent v2 (lightweight tool‑calling)

An interactive CLI agent that auto‑detects a local Ollama model, summarizes PDFs (or general knowledge), and builds a `.pptx` from a single prompt.  
Source: fileciteturn2file1

**Quick start:**
```bash
cd version2
pip install -r requirements.txt
python main.py
```
Try prompts like:
- `make 11 slides ppt about dog`
- `summary only for D:/papers/attention.pdf`
- `read D:/papers/paper.pdf pages 1-3 and summarize`
- `build slides now`

### 🔧 Planned Improvements for v2
Focus areas you flagged:
1) **PDF reading accuracy & speed**
   - Switch/option to **PyMuPDF (`fitz`)** for faster, more robust text extraction (fall back to `pypdf` when needed).
   - Page‑range support and **lazy/streamed parsing** to avoid loading entire files in memory.
   - **Chunking & caching**: intelligent chunk sizes; cache parsed text keyed by file hash & page ranges.
   - Optional **parallel parsing** (multiprocessing) on multi‑core machines.
2) **PPT beautify**
   - Theme presets (corporate/academic/minimal), slide master templates, brand colors & fonts.
   - Better layout balancing (title vs bullets), auto‑fit, consistent spacing, and optional iconography.
   - Optional “References / Further reading” slide.
3) **Quality of life**
   - Clearer progress messages, error handling for scanned PDFs (suggest OCR), and model selection flag.

> 🎯 Goal: Keep v2 “lightweight” but noticeably **faster** on PDFs and **nicer** output decks.

---

## 🕸️ version3 — Agent v3 (LangGraph state machine)

A single‑file agent (`ppt_agent.py`) using **LangGraph** to plan actions: it reads PDFs (with page ranges), generates dense notes, creates a strict JSON outline (title, subtitle, 3–5 bullets per slide), repairs thin/generic slides, and renders a `.pptx`.  
Source: fileciteturn2file0

**Quick start:**
```bash
cd version3
pip install langchain-ollama langgraph langchain-core pydantic pymupdf python-pptx
python ppt_agent.py
```
Prompts you can try:
- `make 11 slides ppt about dog`
- `summary only for "D:/papers/attention.pdf"`
- `read D:/papers/paper.pdf pages 1-3 and summarize`
- `build slides now`
- `regenerate outline`

### ✨ What v3 already does well
- **PyMuPDF** page‑range extraction, **dense notes** generation, **JSON outline with validation**, and **auto‑fixes** for thin or repetitive slides.
- Enforces **Welcome → Content → Conclusion → Thanks** and exact slide counts.
- Smart `python-pptx` rendering with collision‑free filenames.

### 🚧 Planned Improvements for v3
Your requested areas plus advanced feature work:
1) **PDF reading accuracy & speed**
   - Keep PyMuPDF as default; add **adaptive page sampling** for quick “first pass” summaries.
   - **Async / streaming** read path for huge PDFs; optional **background pre‑read** of next pages.
   - Cache: `file_hash + page_ranges` → parsed text; reuse across runs.
2) **PPT beautify**
   - Theming API (preset styles), auto‑layout choice per slide (title‑only, title+bullets, two‑column), and image placeholders.
   - Bullet de‑dup & line‑wrap tuning; consistent font sizes across the deck.
3) **Feature growth**
   - **Summary‑only mode** (produce notes to disk without building slides) exposed as a first‑class command.
   - **Per‑section summaries** (e.g., “summary separately” for selective slides or chapters).
   - **Multi‑document synthesis** (merge notes across many PDFs).
   - **Citations & references** slide from extracted metadata/DOIs.
   - **Retry / self‑repair** when the LLM returns non‑JSON for a slide patch.
   - Optional **image/diagram generation** hooks and media placement.
4) **Developer ergonomics**
   - Config file (`settings.toml`) for defaults (model, theme, slide count).
   - Export intermediate artifacts (`notes.txt`, `outline.json`) to a `./artifacts/` folder.
   - Verbose mode with timing breakdowns for each stage.

> 🎯 Goal: Make v3 the **feature‑rich** line with strong planning (LangGraph), high‑quality decks, and flexible output modes.

---

## 🧪 Repository Layout & Scripts

Suggested folder structure (already used or recommended):
```
version1/
  ppt_builder.py
  README.md
version2/
  main.py
  README.md
version3/
  ppt_agent.py
  README.md
```

Optionally add a top‑level script to run a specific version quickly:
```bash
# Windows Powershell
python .\version3\ppt_agent.py
# macOS/Linux
python ./version3/ppt_agent.py
```

---

## 🧰 Templates & Theming (for v2/v3)
- Add a `/themes` folder with `.potx/.pptx` templates (corporate, academic, minimal).
- Expose `--theme` or `THEME=<name>` to select a template at runtime.
- Provide a default minimalist theme out of the box.

---

## ✅ Acceptance Checklists

**v2 (performance & beautify)**
- [ ] PyMuPDF reader path with page ranges (fallback to pypdf)
- [ ] Parsing cache by file hash + page ranges
- [ ] Faster chunking + optional parallel parse
- [ ] Slide theme presets and improved spacing/auto‑fit
- [ ] References slide (optional)
- [ ] Better progress logs and errors for scanned PDFs

**v3 (features & robustness)**
- [ ] “Summary only” command writes `notes.txt`
- [ ] “Summary separately” per section/page‑range
- [ ] Async/streamed parse for large PDFs
- [ ] Retry/self‑repair for JSON outline and slide densify
- [ ] Theming API + image placeholders
- [ ] Multi‑document synthesis & citations slide
- [ ] Export artifacts (`notes.txt`, `outline.json`) per run

---

## 📝 Licensing

Each version currently references MIT‑style licensing—confirm for your organization and update as needed.

---

## 🙌 Credits

- **version1** pipeline and CLI: slides from topic/PDF with `python-pptx` and local LLM. fileciteturn2file2  
- **version2** lightweight agent with tool‑calling and auto model detection. fileciteturn2file1  
- **version3** LangGraph state machine with planner, robust outline JSON, and densification utilities. fileciteturn2file0
