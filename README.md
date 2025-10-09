# 🧠 PPT Builder — AI-Generated PowerPoint Decks from Research Papers

`ppt_builder.py` is a command-line tool that automatically **creates professional PowerPoint presentations** from **research papers (PDFs)** using **LangChain + Ollama**.  
It extracts key ideas, summarizes them, and designs a slide deck with titles, bullet points, and speaker notes — complete with a welcome and thank-you slide.

---

## 🚀 Features

- 📄 **Reads academic PDFs** and extracts key sections (no OCR required).  
- 🧩 **Uses Ollama local LLMs** (e.g. `llama3.1:8b`, `mistral`, `qwen2`) for summarization and deck generation.  
- 🎨 **Auto-generates PowerPoint slides** with formatted titles, subtitles, and bullets using `python-pptx`.  
- 🔗 **Adds references slide** automatically (optional).  
- ⚙️ **CLI options** for topic, tone, model, and slide count.  
- 🪶 **Smart file naming** using Ollama-generated short names.

---

## 🧰 Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally  
  ```bash
  ollama serve
  ```
- Local model such as `llama3.1:8b` downloaded:
  ```bash
  ollama pull llama3.1:8b
  ```
- Python packages:
  ```bash
  pip install langchain-core langchain-text-splitters langchain-ollama pydantic pypdf python-pptx requests
  ```

---

## 📦 Usage

### 1️⃣ Basic Usage (from topic only)
```bash
python ppt_builder.py --topic "AI Bias in Language Models"
```

### 2️⃣ From a Research Paper
Place your PDF in the same folder, then run:
```bash
python ppt_builder.py --topic "Style Bias in Inversion Models"
```
If multiple PDFs exist, it automatically picks the first one found.  
To specify a file:
```bash
python ppt_builder.py --topic "Inversion Models" --pdf path/to/paper.pdf
```

### 3️⃣ Options
| Flag | Description | Default |
|------|--------------|----------|
| `--topic` | Presentation topic | *(required)* |
| `--audience` | Target audience | `general` |
| `--tone` | Tone of writing | `informative` |
| `--slides` | Number of main content slides (excl. intro/outro) | `8` |
| `--model` | Ollama model name | `llama3.1:8b` |
| `--outfile` | Output PPTX file name | Auto-generated |
| `--template` | Path to PowerPoint theme/template (.potx/.pptx) | None |
| `--pdf` | Path to PDF to analyze | Auto-detects |
| `--no-references` | Disable “References” slide | False |

---

## 🧠 How It Works

1. **PDF Parsing** — Reads all pages using `pypdf` and extracts text.  
2. **Chunking** — Splits text into small segments using LangChain’s `RecursiveCharacterTextSplitter`.  
3. **Summarization** — Each chunk is summarized into key bullet points via the LLM.  
4. **Reduction** — Combines and condenses all notes into 30–40 focused points.  
5. **Deck Generation** — LLM designs a `Deck` schema (topic, tone, slides, etc.).  
6. **PowerPoint Rendering** — Slides are built with `python-pptx`, styled automatically.

---

## 🧩 Example Output

Running:
```bash
python ppt_builder.py --topic "Transformers in Vision Tasks" --pdf paper.pdf
```

Produces:
```
📄 Using PDF: paper.pdf
🪶 Ollama named your file: transformers_in_vision_20251008_123456.pptx
✅ PowerPoint created: transformers_in_vision_20251008_123456.pptx
```

Slides include:
- Welcome slide  
- 8 content slides (Hook → Method → Results → Limitations → Next Steps)  
- References slide (auto-extracted)  
- Thank You slide  

---

## 🧑‍💻 Example Workflow

1. Start Ollama:
   ```bash
   ollama serve
   ```
2. Run the script with your paper:
   ```bash
   python ppt_builder.py --topic "Self-Supervised Learning" --pdf mypaper.pdf
   ```
3. Open the generated `.pptx` file in PowerPoint or Keynote.

---

## 📚 Project Structure

```
ppt_builder.py   # Main script
README.md        # This file
```

---

## 🪪 License

MIT License © 2025

---

## 💡 Tips

- For better results, use a capable local model (`llama3.1:70b`, `mistral-nemo`, etc.).
- You can plug in your own PowerPoint template (`--template mytheme.potx`).
- Works offline — no API keys required.
