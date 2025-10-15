# Version 4 — CPU‑First LangGraph PPT Agent

**Status:** actively working on **CPU‑only model running** and planning a **future user interface**. This version keeps the agentic workflow from earlier versions, but swaps to a **Hugging Face Transformers** stack optimized for CPU execution while we iterate on speed and memory use.

---

## ✨ What v4 is
- A LangGraph‑orchestrated **AI agent** that does:
  1) **Read PDFs** (PyMuPDF) with page‑range support  
  2) **Summarize** (map→reduce) with numeric detail preservation  
  3) **Plan** a strict **slide outline JSON** (titles, subtitles, 3–5 details)  
  4) **Render** a clean PowerPoint via `python-pptx`
- Focused on **running the model on CPU** today; **UI/UX** (desktop/web) comes next.

---

## 🧱 Architecture (single file)

```
version4/
└─ ppt_agent.py     # tools + LangGraph state machine + CLI
```

**Key pieces**  
- **Transformers LLM via pipeline:** `AutoModelForCausalLM` + `AutoTokenizer` → `pipeline("text-generation")` → `HuggingFacePipeline` → `ChatHuggingFace`.  
- **Tools:**  
  - `read_pdf(path, pages?, mode?, ...)` — fast text extraction with PyMuPDF  
  - `summarize_text(text, style, target_words)` — map→reduce; preserves numbers; fallback logic  
  - `slide_outline_json(notes|topic, slide_target)` — JSON schema; validates counts; cleans thin/generic slides; injects quantitative bullets  
  - `build_ppt(outline_json, path)` — generates `.pptx` with safe file naming  
- **Agent graph:** `planner_node` (decide next action) → `tool_node` (run one tool & post‑process) → `END`.  
- **CLI:** single‑turn invoke per prompt; examples printed at startup.

---

## 🖥️ CPU‑Only: How it runs

This build prefers CPU execution out of the box:

- Default model: `HF_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"` (override with env var).  
- **Quantization OFF on CPU**: set `HF_LOAD_4BIT=0` (default). 4‑bit via bitsandbytes is primarily for CUDA; stick to full‑precision CPU for now.  
- Device mapping is `device_map="auto"`; on a CPU‑only host, it stays on CPU.  
- If your model repo ships an internal **MXFP4 quant config**, the loader detects it and chooses a safe dtype automatically; otherwise we fall back to CPU‑friendly float32.  
- Token budget: `HF_MAX_NEW_TOKENS` controls generation length (defaults to the function’s `num_predict`).

**Practical tips**  
- Prefer **≤7B** instruction‑tuned models for comfort on typical CPUs.  
- Close other heavy apps; CPU summarization is compute‑bound.  
- If you have plenty of RAM, you can try 13B models—expect slower runs.

---

## 📦 Requirements

- Python 3.10+ recommended
- `transformers`, `torch`, `langchain-huggingface`, `langgraph`, `langchain-core`
- `pydantic`, `pymupdf` (PyMuPDF), `python-pptx`

Install (example):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install transformers torch langchain-huggingface langgraph langchain-core pydantic pymupdf python-pptx
```

---

## ⚙️ Configuration (env vars)

```bash
# Select the model (CPU‑friendly 7B is a good start)
set HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct      # Windows (cmd)
export HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct   # macOS/Linux

# CPU path: keep 4‑bit off (bitsandbytes typically requires CUDA)
set HF_LOAD_4BIT=0
export HF_LOAD_4BIT=0

# Control output tokens (optional)
set HF_MAX_NEW_TOKENS=600
export HF_MAX_NEW_TOKENS=600
```

Summary controls (optional):
```bash
# Source clamp (0 = unlimited), chunk size, and max chunks for map→reduce
export SUMMARY_SOURCE_CAP=0
export SUMMARY_CHUNK_SIZE=3000
export SUMMARY_MAX_CHUNKS=0
```

---

## 🚀 Usage

```bash
python ppt_agent.py
```

You can say:
- `summary only for "D:/papers/attention.pdf"`
- `read "D:/papers/paper.pdf" pages 1-3 and summarize`
- `make 10 slides ppt about diffusion models`
- `build slides now`
- `regenerate outline`

**Flow examples**
- **Summary‑only:** read → summarize → finish (no PPT)  
- **Topic → deck:** outline → build  
- **PDF → deck:** read → summarize → outline → build (or `build slides now` to auto‑chain)

---

## 🧠 Outline & Quality Controls

- Enforces exact slide count (default **11**, or parsed from “make 14 slides …”).  
- **Welcome / Conclusion / Thanks** structure guaranteed.  
- **Numeric facts** are pulled from notes and injected across slides when needed.  
- Cleans generic titles (e.g., “Topic 1”), removes duplicate bullets, and ensures **3–5** details per content slide.  
- If a slide is thin, it synthesizes details from notes; on failures, it attempts an LLM densify pass, then deterministic fallback.

---

## 🎯 Roadmap (near‑term)

- **CPU model running (ongoing):**
  - Tune chunk sizes, reduce memory spikes, and provide timing breakdowns.
  - Optional **sentence‑window map** for faster summaries on long PDFs.
  - Smarter early‑stop during PDF read (chars/pages caps already supported).

- **UI/UX (next):**
  - Minimal desktop UI (PySide/Qt) or a small web UI.
  - File picker, page‑range helper, theme selection, live progress, and a “summary‑only” export button.

- **PPT beautify:**
  - Themes & templates (`.potx`) and balanced layouts (two‑column, title‑only).
  - Consistent font sizing & spacing; optional references slide.

- **Feature growth:**
  - Multi‑document synthesis; citations slide; image/diagram placeholders.
  - Config file (`settings.toml`) for default model/theme/slide count.

---

## 🛠 Troubleshooting

- **Slow on CPU** → Use a 7B model; clamp `SUMMARY_CHUNK_SIZE`, consider fewer `SUMMARY_MAX_CHUNKS` during testing.  
- **Out of memory** → Close apps; try a smaller model; shorten PDFs via page ranges (`pages="1-10"`).  
- **Empty or weird text from PDF** → It might be scanned; perform OCR first.  
- **4‑bit quant failed** → Keep `HF_LOAD_4BIT=0` on CPU‑only machines.

---

## 📄 License

MIT (example) — adjust to your needs.
