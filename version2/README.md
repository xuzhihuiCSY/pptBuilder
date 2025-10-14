# LangGraph PPT Agent version2

A lightweight LangChain + Ollama command‑line agent that searches local PDFs (or general knowledge), summarizes content, and auto‑builds a PowerPoint deck from a single prompt.

---

## ✨ Features

- Auto-detects local **Ollama** models and uses the first available one.
- Pluggable tools for:
  - Listing PDFs in a directory
  - Summarizing PDFs
  - Generating knowledge-based summaries (when no PDF is found)
  - Creating a PowerPoint presentation from the summary
- Single interactive loop: type a request; get a `.pptx`.

> **Entrypoint:** `main.py`

---

## 🧱 Project Structure (minimal)

```
.
├─ main.py          # CLI entrypoint: agent wiring, prompt, run loop
└─ (your tools)     # Optional: modules used by the agent (e.g., pdf, knowledge, pptx helpers)
```

---

## 📦 Requirements

- **Python 3.9+**
- **Ollama** running locally (default: `http://localhost:11434`)
  - Install: https://ollama.com
  - Pull at least one model (e.g. `ollama pull llama3.1:8b`)
- **PowerPoint export dependency** (typically `python-pptx`) if your `create_powerpoint` tool uses it
- **Optional**: `TAVILY_API_KEY` if your summarization uses web search

---

## 🔧 Installation

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2) Install project dependencies
pip install -r requirements.txt
```

Make sure Ollama is running (usually `http://localhost:11434`):

```bash
ollama serve
# and pull at least one model
ollama pull llama3.1:8b
```

---

## 🚀 Usage

Start the agent CLI:

```bash
python main.py
```

You’ll see a banner like:

```
🔎 Checking for local Ollama models...
✅ Using model: llama3.1:8b
🤖 PowerPoint Agent is ready. Let's create a presentation! (type 'exit' to quit)
```

Now type natural-language requests, for example:

- `make 11 slides ppt about dog`
- `summary only for D:/papers/attention.pdf`
- `read D:/papers/paper.pdf pages 1-3 and summarize`
- `build slides now`

The agent will:
1) Look for relevant PDFs,
2) Summarize a match or fall back to knowledge,
3) Build slides,
4) Print where the `.pptx` was saved.

---

## ⚙️ Configuration

- **Model selection**: auto-selects the first model returned by Ollama’s `/api/tags`.
- **Propagation**: tools that call an LLM should receive the same `model_name` the agent is using.
- **Environment variables**:
  - `TAVILY_API_KEY` *(optional)* — for web-search–backed summarization flows.
- **Paths**: you can reference absolute paths in prompts (e.g., `D:/papers/attention.pdf`).

---

## 🧪 Example Session

```
You: make 10 slides about diffusion models
Agent: Found diffusion_models_tutorial.pdf → summarized → created Diffusion_Models.pptx at D:\work_on_github\...\Diffusion_Models.pptx
```

---

## 🛠 Troubleshooting

- **Cannot connect to Ollama / exits immediately**
  - Ensure `ollama serve` is running locally.
  - Verify models exist: `ollama pull llama3.1:8b`.
- **“No model found / agent exited”**
  - Pull at least one model and rerun.
- **PowerPoint issues**
  - Confirm `python-pptx` is installed and any template/resources exist.

---

## 📚 Extending

- Add tools (diagram/image generation, code summarization, etc.).
- Swap or pin models; adjust `temperature` in the LLM.
- Add slide templates/themes and branding support.
- Multi-document synthesis and citation slides.

---

## 📄 License

MIT (example). Update this section to match your chosen license.
