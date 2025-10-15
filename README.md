# PPT Builder & Agents — Monorepo README (v1 · v2 · v3 · v4)

This repository contains four related tools for turning topics and PDFs into PowerPoint decks. They evolve from a classic, non-agent pipeline (**v1**) to increasingly agentic systems (**v2**, **v3**) and a CPU-first agent with a future UI plan (**v4**).

```
.
├─ version1/   # Classic builder (non-agent) — stable
├─ version2/   # Lightweight AI agent (tool-calling)
├─ version3/   # LangGraph AI agent (planner + repairs)
└─ version4/   # CPU-first LangGraph AI agent (UI planned)
```

---

## 🧭 Which version should I pick?

| Scenario | Recommended version |
|---|---|
| I want a simple, predictable CLI that converts a topic or PDF to slides with minimal moving parts. | **v1 (non-agent, stable)** |
| I want a prompt-driven CLI that uses a local LLM to call tools and build decks. | **v2 (lightweight agent)** |
| I want a more robust agent with planning, JSON-outline validation, and self-repair for slides. | **v3 (LangGraph agent)** |
| I need CPU-only model running today, and I’d like a future UI (desktop/web) on top. | **v4 (CPU-first LangGraph agent)** |

**Agent or not?**
- **v1**: not an agent (fixed pipeline; may call LLM once for summary).
- **v2**: yes—lightweight agent (prompt → tool calls → PPT).
- **v3**: yes—LangGraph agent (planner + validation + repairs).
- **v4**: yes—LangGraph agent focused on CPU-only models; UI coming next.

---

## ✅ Common Requirements

- **Python** 3.9+ (3.10+ recommended for v4)
- **PowerPoint rendering:** `python-pptx`
- **PDF reading:** `pymupdf` (PyMuPDF) recommended for v2–v4, `pypdf` is fine for v1
- **Local LLMs:**
  - **v1–v3:** typically Ollama-backed chat models (e.g., `llama3.1:8b`)
  - **v4:** Hugging Face Transformers on CPU (`torch`, `transformers`, etc.)

Create a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## 📁 Version Overviews

### version1 — Classic Builder (non-agent · stable)
- **What it does:** Deterministic pipeline → read PDF/topic → summarize → build PPT.
- **Why use it:** Minimal dependencies, predictable outputs, no planning layer.
- **Status:** Feature-frozen; good baseline and for constrained environments.

**Install (example):**
```bash
pip install langchain-core langchain-text-splitters langchain-ollama pydantic pypdf python-pptx requests
```

**Run:**
```bash
# Topic only
python ppt_builder.py --topic "AI Bias in Language Models"

# From a specific PDF
python ppt_builder.py --topic "Inversion Models" --pdf path/to/paper.pdf
```

---

### version2 — Lightweight AI Agent (tool-calling)
- **What it does:** Prompt → agent picks tools → summarize PDF or topic → build PPT.
- **How you use it:** Conversational commands like `make 11 slides ppt about dog`, `summary only for D:/paper.pdf`, `build slides now`.
- **Focus for improvements:** PDF reading **accuracy & speed**, **PPT beautify**.

**Install & run:**
```bash
cd version2
pip install -r requirements.txt
python main.py
```

**Planned enhancements (v2):**
- PyMuPDF reader path + page ranges; fallback to `pypdf`.
- Parsing cache by file hash + page ranges; optional parallel parsing.
- Theme presets, improved spacing/auto-fit, references slide.
- Better progress messages and model-selection flags.

---

### version3 — LangGraph AI Agent (planner + validation + repairs)
- **What it does:** LangGraph chooses steps (read → summarize → outline → build); validates JSON outline; fixes thin/generic slides; enforces **Welcome → Content → Conclusion → Thanks** and exact slide counts.
- **Why it’s “more agentic”:** It plans, validates, and repairs with tool calls and caches artifacts.

**Install & run:**
```bash
cd version3
pip install langchain-ollama langgraph langchain-core pydantic pymupdf python-pptx
python ppt_agent.py
```

**Useful commands:**
- `summary only for "D:/papers/attention.pdf"`
- `read D:/papers/paper.pdf pages 1-3 and summarize`
- `make 11 slides ppt about diffusion models`
- `build slides now`
- `regenerate outline`

**Planned enhancements (v3):**
- Faster/async PDF reading; adaptive page sampling; better caching.
- Theming API, two-column/title-only layouts, image placeholders.
- “Summary-only” & “summary separately” (per section/page range).
- Multi-document synthesis; citations/reference slide.
- Export artifacts (`notes.txt`, `outline.json`) with timing breakdowns.

---

### version4 — CPU-First LangGraph AI Agent (UI planned)
- **What it is now:** LangGraph agent using **Hugging Face Transformers** on **CPU only**; optimizing for speed/memory on CPU.
- **What’s next:** A simple **UI** (desktop/web) for file-picking, page-range helpers, theme selection, progress, and “summary-only” export.

**Install & run:**
```bash
cd version4
pip install transformers torch langchain-huggingface langgraph langchain-core pydantic pymupdf python-pptx
python ppt_agent.py
```

**CPU tips:**
- Start with ≤7B instruction-tuned models (e.g., `Qwen2.5-7B-Instruct`).
- Keep `HF_LOAD_4BIT=0` on CPU-only boxes; bitsandbytes 4-bit is mostly for CUDA.
- Control generation length via `HF_MAX_NEW_TOKENS`.

**Roadmap (v4):**
- CPU: tune chunk sizes; reduce memory spikes; timing breakdowns.
- UI: PySide/Qt or small web UI; templates/themes; live progress.
- Beautify: `.potx` themes; balanced layouts; references slide.
- Features: multi-doc synthesis; citations; image/diagram hooks; `settings.toml` config.

---

## 🧰 Templates & Theming (v2–v4)
- Provide a `/themes` folder with `.potx/.pptx` templates (corporate, academic, minimalist).
- Add a `--theme` or `THEME=<name>` flag for quick selection.
- Keep a clean default theme for out-of-the-box usage.

---

## 🧪 Repository Structure (suggested)

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
version4/
  ppt_agent.py
  README.md
themes/           # optional templates
```

---

## 🛠 Troubleshooting (common)

- **Slow summaries** → Use smaller models (7B), clamp PDF page ranges, reduce chunk size.  
- **Empty PDF text** → Likely scanned; OCR first.  
- **Out-of-memory** → Close apps, use smaller model, shrink page ranges or chunk sizes.  
- **Non-JSON outline (v3/v4)** → Retry; switch model; the agents attempt self-repair/fallbacks.

