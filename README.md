# ⚖️ Legal/Policy Document Q&A (RAG)

> RAG-powered Q&A system for legal and policy documents. Upload contracts, privacy policies, or terms of service — ask questions, get cited answers from the source text.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)

## Overview

This application uses **Retrieval-Augmented Generation (RAG)** to provide accurate, source-cited answers to questions about legal documents. Upload any legal document (PDF, DOCX, TXT), and the system will parse it, create searchable embeddings, and answer your questions with citations pointing to the exact source passages.

### Key Features

- **Multi-format support** — Upload PDF, DOCX, or TXT documents
- **Semantic search** — Find relevant passages by meaning, not just keywords
- **Cited answers** — Every answer includes source citations with page numbers
- **Multi-turn conversations** — Ask follow-up questions with context memory
- **Multi-document support** — Upload multiple documents and search across all or filter by specific ones
- **Document management** — List, delete, and re-upload documents
- **Duplicate detection** — Prevents accidental re-uploads
- **Chat UI** — Streamlit-based interface with document management sidebar

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini API (swappable to Anthropic Claude) |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Document Processing | pypdf, python-docx |

## Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit   │────▶│   RAG Chain   │────▶│  Gemini API  │
│     UI       │     │              │     │  (LLM)       │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │ Vector Store  │
                    │  (ChromaDB)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Embeddings   │
                    │ (MiniLM-L6)   │
                    └──────────────┘
```

## Quick Start

```bash
# Clone & setup
git clone https://github.com/SahilMukadam/Legal_Policy_Document_Q-A_RAG.git
cd Legal_Policy_Document_Q-A_RAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the Streamlit UI
streamlit run app.py

# OR run the FastAPI backend
uvicorn src.api.main:app --reload --port 8000
```

## Usage

### Streamlit UI (Recommended)
```bash
streamlit run app.py
```
1. Upload documents via the sidebar
2. Ask questions in the chat interface
3. View cited answers with expandable source passages
4. Filter by specific documents using the sidebar dropdown

### REST API
```bash
uvicorn src.api.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive API docs
```

**Endpoints:**
- `POST /upload` — Upload a document
- `POST /ask` — Ask a question (with citations)
- `POST /search` — Semantic search
- `GET /documents` — List all documents
- `DELETE /documents/{filename}` — Delete a document
- `GET /history/{session_id}` — Get conversation history
- `GET /stats` — Vector store statistics

## Project Structure

```
├── app.py                    # Streamlit frontend
├── src/
│   ├── api/main.py           # FastAPI backend
│   ├── ingestion/
│   │   ├── parser.py         # Document parsing (PDF, DOCX, TXT)
│   │   └── chunker.py        # Text chunking with overlap
│   ├── embeddings/
│   │   └── embedding_service.py  # Sentence-Transformers embeddings
│   ├── retrieval/
│   │   └── vector_store.py   # ChromaDB vector storage & search
│   ├── chains/
│   │   └── rag_chain.py      # RAG pipeline with conversation memory
│   └── llm_provider.py       # Multi-provider LLM factory
├── configs/settings.py       # Environment-based configuration
├── tests/                    # Comprehensive test suite
├── notebooks/                # Demo scripts
└── data/sample_docs/         # Sample legal documents
```

## Switching LLM Providers

The project uses a provider factory pattern. To switch from Gemini to Claude:

```bash
# In .env, change:
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
```

No code changes needed.

## Running Tests

```bash
pytest tests/ -v
```

## Features Checklist

- [x] Document ingestion (PDF, DOCX, TXT)
- [x] Text chunking with configurable strategies
- [x] Embedding generation & ChromaDB storage
- [x] Semantic search with metadata filtering
- [x] RAG chain with LLM integration
- [x] Citation system (source + page numbers)
- [x] Conversation memory (multi-turn Q&A)
- [x] Multi-document support
- [x] Document management (list, delete, duplicate detection)
- [x] Streamlit chat UI
- [ ] Pinecone cloud migration
- [ ] Evaluation pipeline
- [ ] Hybrid search + re-ranking

---

*Part of a 4-project AI/ML portfolio. See also: [Smart City Data Agent](#), [Code Instruction Fine-tuner](#), [LLM Eval Benchmark](#).*
