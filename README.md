---
title: CyberTA
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
license: mit
---

# 🤖 CyberTA – AI Teaching Assistant

A fully open-source RAG (Retrieval-Augmented Generation) pipeline wrapped in a
Gradio UI. Upload course PDFs, DOCX, or PPTX files → visualise how concepts
cluster in vector space → chat with an AI that answers **only from your
documents**.

---

## Features

| Feature | Details |
|---|---|
| **Document ingestion** | PDF · DOCX · PPTX with chunking & overlap |
| **Open-source embeddings** | MiniLM, MPNet, BGE, E5, GTE, Nomic (all local via `sentence-transformers`) |
| **Vector store** | ChromaDB (persistent, local) |
| **Vector visualisation** | PCA & UMAP → interactive 2D + 3D Plotly charts, coloured by source |
| **LLM choices** | HuggingFace Inference API · Ollama (local) · OpenAI · Anthropic |
| **Models available** | Gemma 1B/4B · Qwen2.5 1.5B/7B · Llama3.2 1B/3B · Mistral 7B · Phi-3.5 · GPT-4o-mini · Claude Sonnet |

---

## Local Setup

```bash
git clone https://github.com/<you>/CyberTA
cd CyberTA
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → open http://localhost:7860
```

### Ollama (local models, no API key needed)

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2:1b      # ~800 MB – fastest
ollama pull llama3.2:3b      # ~2 GB
ollama pull gemma3:1b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull phi3.5
ollama pull mistral
ollama serve                 # keeps running in the background
```

Then in the **⚙️ Model Setup** tab, leave `OLLAMA_HOST` as `http://localhost:11434`.

---

## Deploying to Hugging Face Spaces

### 1 – Create the Space

1. Go to **huggingface.co → New Space**
2. Choose **SDK: Gradio**, hardware **CPU Basic** (free) or **CPU Upgrade** for larger models
3. Set visibility to *Public* or *Private*

### 2 – Push the code

```bash
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/CyberTA
git push hf main
```

### 3 – Set Secrets (API keys) — never commit keys to git

Go to your Space → **Settings → Variables and Secrets → New Secret**:

| Secret name | When needed | Where to get it |
|---|---|---|
| `HF_TOKEN` | **Required for HF models** | huggingface.co → Settings → Access Tokens → New token (role: *Read*) |
| `OPENAI_API_KEY` | Only for GPT models | platform.openai.com → API Keys |
| `ANTHROPIC_API_KEY` | Only for Claude | console.anthropic.com → API Keys |

> Ollama cannot run on HF Spaces (no local server). Use HF Inference API models there.

### 4 – Gated models on HuggingFace (Llama, Gemma)

Some models require you to **accept a licence** before the API allows access:

- **Llama 3.2** → [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) → click **"Agree and access"**
- **Gemma 3** → [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) → click **"Acknowledge licence"**

You only need to do this once per account. Your `HF_TOKEN` then grants access automatically.

---

## Architecture

```
User uploads PDF/DOCX/PPTX
        │
        ▼
  Text Extraction  (PyMuPDF / python-docx / python-pptx)
        │
        ▼
   Chunking  (500 chars, 80 overlap)
        │
        ▼
  Embedding  (sentence-transformers, runs locally)
        │
        ▼
  ChromaDB  (cosine similarity index, persisted to ./chroma_db)
        │
  ┌─────┴──────┐
  │            │
  ▼            ▼
Visualise   RAG Query
(PCA/UMAP)  embed question → retrieve top-k → build prompt → LLM → answer
```

---

## File Structure

```
CyberTA/
├── app.py            # Gradio UI (4 tabs)
├── rag_pipeline.py   # Ingestion, embedding, ChromaDB, retrieval
├── llm_providers.py  # HuggingFace / Ollama / OpenAI / Anthropic routing
├── visualize.py      # PCA/UMAP 2D+3D Plotly figures
├── requirements.txt
├── .gitignore
└── README.md
```

---

## License

MIT
