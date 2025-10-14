# LangGraph PPT Agent (ppt_agent.py) version 3

An end‑to‑end **research → notes → outline → PowerPoint** agent built with **LangGraph**, **LangChain + Ollama**, **PyMuPDF**, and **python‑pptx**. It reads local PDFs (optionally by page range), generates dense slide‑ready notes, structures them into a JSON slide outline, and renders a `.pptx`—all from a conversational CLI.

> This README documents `ppt_agent.py` (single‑file version with tools + graph + CLI).

---

## ✨ Highlights

- **PDF ingestion** using PyMuPDF (`fitz`) with page‑range support like `1-3,5`.
- **Dense notes generation** via your local **Ollama** model (`ChatOllama`).
- **Robust slide outline** creation:
  - Enforces a strict JSON schema (title, subtitle, 3–5 details).
  - Cleans generic titles, dedups repetitive bullets, and spreads unique details across slides.
  - Auto‑fixes thin slides by synthesizing facts from notes or calling the LLM to densify a slide.
  - Guarantees **Welcome → Content → Conclusion → Thanks** structure and exact slide count.
- **PPT renderer** built on `python-pptx`:
  - Smart filename from deck title; auto‑dedups (`Title_2.pptx`, …).
  - Welcome and Thanks slides have no bullets; Conclusion enforces 3–5 takeaways.
- **LangGraph state machine** with a planner that decides the next action (read, summarize, outline, build, chat, finish) and a tool node that executes it.
- **CLI workflow** with natural commands like `summary only`, `build slides now`, `regenerate outline`, or “make 11 slides ppt about dog”.

Source: fileciteturn1file0

---

## 🧱 Architecture Overview

```
ppt_agent.py
├─ Tools
│  ├─ read_pdf(path, pages?)                # PyMuPDF text extraction (soft length caps)
│  ├─ summarize_text(text, style, target)   # ChatOllama → dense slide notes
│  ├─ slide_outline_json(notes, slide_target)
│  │     ├─ densify_slide_with_llm(...)     # Improve single slide via JSON-only prompt
│  │     ├─ polish_outline(...), allocate_details_across_slides(...)
│  │     └─ helpers: keyphrase titles, de-dup, sentence split, etc.
│  └─ build_ppt(outline_json, path?)        # python-pptx rendering
│
├─ LangGraph
│  ├─ planner_node(...)   # Chooses next action (JSON), with heuristics fallback
│  ├─ tool_node(...)      # Executes chosen tool; caches artifacts
│  ├─ chat_node(...)      # Lightweight non-tool replies
│  └─ build_app()         # Wires graph and edges
│
└─ CLI main
   ├─ Prints examples
   ├─ Single-turn invoke of graph per user message
   └─ Shows final status and PPT path
```

---

## 🔧 Installation

**Requirements** (Python 3.9+ recommended):
- `langchain-ollama`, `langgraph`, `langchain-core`
- `pydantic`
- `PyMuPDF` (`fitz`)
- `python-pptx`

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install langchain-ollama langgraph langchain-core pydantic pymupdf python-pptx
```

Install and start **Ollama**, and pull at least one model:
```bash
# https://ollama.com
ollama serve
ollama pull gpt-oss:20b     # or llama3.1:8b, qwen2.5, etc.
```

> The script defaults to `OLLAMA_MODEL=gpt-oss:20b`. Override via env var:
> ```bash
> # Windows PowerShell
> $env:OLLAMA_MODEL="llama3.1:8b"
> # macOS/Linux
> export OLLAMA_MODEL="llama3.1:8b"
> ```

---

## 🚀 Usage

Run the CLI:
```bash
python ppt_agent.py
```

Sample banner and prompts:
```
🧠 Using Ollama model: llama3.1:8b
📽️ LangGraph PPT Agent ready. Type your request (or 'exit').
Examples:
 - summary only for D:/papers/attention.pdf
 - make slides about diffusion models (10 slides)
 - read D:/papers/paper.pdf pages 1-3 and summarize
 - build slides now
```

Try commands like:
- `make 11 slides ppt about dog`
- `summary only for "D:/papers/attention.pdf"`
- `read D:/papers/paper.pdf pages 1-3 and summarize`
- `build slides now`
- `regenerate outline`

**What happens:**
1) `read_pdf` (if you referenced a `.pdf`) → caches `pdf_text` (soft cap ~30k chars).  
2) `summarize_text` produces ~900–1200 words of dense notes (no bullets).  
3) `slide_outline_json` makes exact‑N slide JSON; fixes thin/generic slides and spreads unique bullets.  
4) `build_ppt` renders a `.pptx`, choosing a safe filename from the deck title and avoiding collisions.  

Artifacts:
- `outline_debug.json` is written when outlining to help with inspection.
- Printed absolute path to the built deck (e.g., `...\Diffusion_Models.pptx`).

---

## ⚙️ Notable Behaviors & Heuristics

- **Slide count**: “make 14 slides …” is parsed automatically; default is **11** if unspecified.
- **“summary only”**: stops after notes generation—no PPT render.
- **“build slides now”**: triggers render if an outline already exists.
- **Page ranges**: `1-3,5` supported; out‑of‑range pages are ignored.
- **Debulleting & dedup**: removes `* - •`, compresses near duplicates (Jaccard), and ensures 3–5 details.
- **Title repair**: generic/bad titles (e.g., “Topic 1”) are replaced by keyphrase‑based candidates.
- **Global uniqueness**: details are distributed to minimize repetition across slides.
- **Fallbacks**: if the LLM returns non‑JSON for densification, deterministic synthesis from notes is used.

---

## 🛠 Troubleshooting

- **No models found / can’t connect to Ollama**  
  Ensure `ollama serve` is running and you’ve pulled a model (`ollama pull llama3.1:8b` or `gpt-oss:20b`).

- **Empty or short slides**  
  Check that your PDF has extractable text. Scanned PDFs may need OCR beforehand.

- **Non‑JSON from LLM in outline/densify**  
  The code attempts to recover; if still off, try a different model or re‑run.

- **Long PDFs truncation**  
  `read_pdf` caps extracted text (~25–30k chars) to keep the token budget manageable.

- **Windows paths**  
  The planner can parse quoted or unquoted Windows paths ending with `.pdf`.

---

## 🧩 Extensibility Ideas

- Slide theming & corporate branding (python‑pptx templates).
- Web retrieval augmentation (add search tool).
- Image/diagram generation tool and media placement per slide.
- Persistent vector store for multi‑doc synthesis.
- Rich CLI flags (e.g., `--slides 15 --out MyDeck.pptx --theme corporate`).

---

## 📄 License

MIT (example). Adjust to your needs.
