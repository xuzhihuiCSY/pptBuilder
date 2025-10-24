# PPT Agent (LangGraph + Ollama) version 6

A local-first presentation agent that uses **LangGraph** (ReAct) and **Ollama** to:
- Read PDFs or text files, summarize them, generate an outline, and build a **PowerPoint (.pptx)**.
- Build a new deck directly from a topic.
- Describe the latest deck without rebuilding it.
- Run with a simple **Kivy** GUI launcher (`kivy_runner.py`) or in the terminal.

> Core agent: `ppt_agent.py`  
> GUI runner: `kivy_runner.py`

---

## 1) Requirements

- Python **3.11+** recommended
- [Ollama](https://ollama.ai) with a local model (default in this project: `gpt-oss:20b`)
- System packages:
  ```bash
  pip install "langchain>=0.3.0" "langgraph>=0.2.20" "langchain-community>=0.3.0" "langchain-ollama>=0.1.0" pypdf python-pptx
  ```
- Pull the Ollama model:
  ```bash
  ollama pull gpt-oss:20b
  ```

> The agent renders slides using **python-pptx** and persists state in `agent_state.json` (topic, last outline, last text/summary, last PPT path). Deck files are written to the `./decks/` folder.

---

## 2) Quick Start (Terminal)

From the project directory:

```bash
python ppt_agent.py --verbose
```

You’ll see a REPL like:

```
ppt_agent (LangGraph ReAct + Ollama) — type your request. Ctrl+C to exit.
Model: gpt-oss:20b
user > 
```

### Examples
- Build from topic:
  ```
  user > build a ppt about large language models
  ```

- Read a PDF then build:
  ```
  user > read pdf D:\path\to\paper.pdf
  user > build ppt based on the paper
  ```

- Ask about the latest deck (no rebuild):
  ```
  user > tell me about the ppt
  ```

### Flags
- `--verbose` — show **step-by-step** agent and tool activity.
- `--recursion N` — allow more agent steps per turn (default 12). If the model needs more thinking:
  ```bash
  python ppt_agent.py --verbose --recursion 24
  ```
- `--slides N` — default slide count (title + content slides).

---

## 3) Kivy GUI Runner

Launch a small GUI console to start/stop the agent and send commands:

```bash
python kivy_runner.py
```

- **Start** launches `ppt_agent.py --verbose` in a child process.
- **Stop** sends a graceful signal (CTRL+BREAK on Windows / SIGINT on macOS/Linux), with terminate/kill fallback.
- Type commands in the bottom input and press **Enter** or **Send**.
- Output appears in the large console view.

> The GUI runner uses a Windows-friendly process model (new process group) and stops without producing EOF errors in the agent.

---

## 4) How It Works

- **Tools**:
  - `read_pdf(path)` — reads PDF/txt/md and stores `last_text`.  
  - `summarize_text(text)` — quick extractive summary; stores `last_summary`.
  - `slide_outline_json(topic, notes)` — calls your **Ollama** model to produce a **strict JSON** outline.
  - `build_ppt(outline_json)` — renders `.pptx` with `python-pptx` (accepts str or dict).
  - `describe_latest_deck()` — lists the latest deck path, topic, and a few slide titles.

- **Autonomy**
  - For phrases like “**build … ppt … about …**” the agent calls: _outline → build_.
  - For “**based on this paper**” it uses your saved summary/text from `read_pdf`/`summarize_text`.
  - If the LLM stalls (no tool call), a **safe fallback** runs _outline → build_ once.

- **Persistence**
  - State is stored in `agent_state.json`: `topic`, `last_outline`, `last_text`, `last_summary`, `ppt_path`.

- **Output**
  - Decks are saved under `./decks/` with a sanitized filename (title + timestamp).

---

## 5) Troubleshooting

- **Model not found**  
  Ensure Ollama has the model:
  ```bash
  ollama pull gpt-oss:20b
  ```
  You can change the default model by setting the env var:
  ```bash
  set OLLAMA_MODEL=llama3.1:8b-instruct    # Windows (cmd)
  export OLLAMA_MODEL=llama3.1:8b-instruct # macOS/Linux (bash/zsh)
  ```

- **Too few agent steps**  
  Increase recursion steps:
  ```bash
  python ppt_agent.py --verbose --recursion 24
  ```

- **EOF after Stop**  
  The GUI runner avoids stdin-close and uses signals. If you still run in terminal, add EOF guard to `ppt_agent.py` loop:
  ```python
  try:
      user_text = input("user > ").strip()
  except EOFError:
      print("\nBye."); break
  ```

- **Docstring error for @tool**  
  Every `@tool` function needs a proper docstring. The provided files include them.

- **Validation error: build_ppt expects a string**  
  The patched `build_ppt` accepts either **str or dict**; replace if you see this error in an older copy.

---

## 6) Folder Layout

```
version6/
├─ ppt_agent.py        # CLI agent (LangGraph + Ollama)
├─ kivy_runner.py      # Kivy GUI to start/stop the agent
├─ decks/              # Output PPTXs
└─ agent_state.json    # Agent memory (created at runtime)
```

---

## 7) Security & Privacy

- All processing runs **locally**.
- PDFs and decks stay on your machine.
- No calls to external APIs beyond your local Ollama server.

---

## 8) License

MIT (or your project’s license).

---

## 9) Credits

- Built on: **LangChain**, **LangGraph**, **Ollama**, **python-pptx**, **pypdf**.
- UI with **Kivy**.
