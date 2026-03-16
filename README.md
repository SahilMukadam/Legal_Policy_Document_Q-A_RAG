# ⚖️ Legal/Policy Document Q&A — RAG System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6F00?style=flat)](https://trychroma.com)
[![Tests](https://img.shields.io/badge/Tests-60%2B_passing-4CAF50?style=flat)]()

> Upload legal documents. Ask questions in plain English. Get accurate, cited answers from the source text.

A production-quality **Retrieval-Augmented Generation (RAG)** system built for legal and policy document analysis. Upload contracts, privacy policies, or terms of service — the system parses, embeds, and stores them, then answers your questions with citations pointing to exact source passages.

---

## ✨ Key Features

**Document Intelligence**
- Multi-format support (PDF, DOCX, TXT) with automatic text extraction
- Smart chunking with configurable overlap for optimal retrieval
- Collection-based organization — group documents into folders

**Advanced Search**
- Hybrid search combining semantic (meaning-based) and keyword (BM25) retrieval
- Reciprocal Rank Fusion merges both search methods for best results
- Source filtering — restrict answers to specific documents or collections

**Cited Answers**
- Every answer includes source citations with document name and page number
- Expandable source passages so users can verify answers
- Faithfulness-aware — system only answers from provided context

**Conversation Memory**
- Multi-turn Q&A — ask follow-up questions naturally
- Session-based isolation — multiple users don't mix contexts
- Context-enhanced retrieval — follow-ups search using conversation history

**Production Features**
- Response caching with TTL — repeated questions answered instantly
- Swappable LLM backend (Gemini ↔ Claude, one config change)
- Swappable vector store (ChromaDB ↔ Pinecone, one config change)
- Duplicate detection and document lifecycle management
- Evaluation pipeline measuring retrieval recall, answer correctness, and faithfulness
- 60+ automated tests

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │   Document    │  │  Chat         │  │  Source Citations     │ │
│  │   Management  │  │  Interface    │  │  (expandable)         │ │
│  └──────┬───────┘  └───────┬───────┘  └──────────────────────┘ │
└─────────┼──────────────────┼────────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                          │
│  /upload  /ask  /search  /documents  /collections  /stats       │
└─────────┬──────────────────┬────────────────────────────────────┘
          │                  │
    ┌─────▼─────┐    ┌──────▼──────────────────────────────┐
    │ INGESTION │    │           RAG CHAIN                  │
    │           │    │                                      │
    │ Parser    │    │  Question ──► Hybrid Search           │
    │ (PDF,     │    │              (Semantic + BM25)        │
    │  DOCX,    │    │                  │                    │
    │  TXT)     │    │           Reciprocal Rank Fusion      │
    │     │     │    │                  │                    │
    │ Chunker   │    │         Top-K Context Passages        │
    │ (Recursive│    │                  │                    │
    │  overlap) │    │     ┌────────────▼────────────┐      │
    └─────┬─────┘    │     │    LLM (Gemini/Claude)  │      │
          │          │     │    + System Prompt       │      │
          │          │     │    + Conversation Memory │      │
          │          │     └────────────┬────────────┘      │
          │          │                  │                    │
          │          │          Cited Answer                 │
          │          └──────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────┐
    │       VECTOR STORE             │
    │   (ChromaDB or Pinecone)       │
    │                                │
    │  Embeddings: all-MiniLM-L6-v2  │
    │  384 dimensions, cosine sim    │
    │  Metadata: source, page,       │
    │           collection, chunk_id │
    └────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Google Gemini 2.5 Flash (swappable to Claude) | Free tier, strong at document Q&A |
| **Orchestration** | LangChain | Industry-standard RAG framework |
| **Vector Store** | ChromaDB (local) / Pinecone (cloud) | Swappable via config |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Fast, free, 384-dim vectors |
| **Keyword Search** | Custom BM25 implementation | Catches exact matches semantic misses |
| **Backend** | FastAPI | Async-ready, auto-docs, type-safe |
| **Frontend** | Streamlit | Rapid prototyping, glassmorphism UI |
| **Doc Processing** | pypdf, python-docx | PDF + Word document parsing |
| **Caching** | Custom TTL cache | Avoids redundant LLM calls |
| **Evaluation** | Custom pipeline | Retrieval recall, correctness, faithfulness |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key ([free — get it here](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone
git clone https://github.com/SahilMukadam/Legal_Policy_Document_Q-A_RAG.git
cd Legal_Policy_Document_Q-A_RAG

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env → add your GOOGLE_API_KEY
```

### Run

```bash
# Streamlit UI (recommended)
streamlit run app.py

# OR FastAPI backend only
uvicorn src.api.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive API
```

### Test

```bash
pytest tests/ -v  # 60+ tests
```

---

## 📖 Usage

### Via Streamlit UI

1. **Upload** — Select or create a collection, drop a PDF/DOCX/TXT file
2. **Ask** — Type a question in the chat input
3. **Verify** — Expand "View Sources" to see the exact passages used
4. **Filter** — Use "Search Scope" to restrict answers to specific documents
5. **Follow up** — Ask follow-up questions naturally — the system remembers context

### Via REST API

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload?collection=Leases&force=true" \
  -F "file=@contract.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is rent due?", "k": 5}'

# Search documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "termination clause", "k": 3}'

# List collections
curl http://localhost:8000/collections

# Get stats (includes cache info)
curl http://localhost:8000/stats
```

**Full API docs**: `http://localhost:8000/docs` (Swagger UI)

---

## 🔄 Swappable Providers

Change one line in `.env` — zero code changes:

```bash
# LLM: Switch between Gemini (free) and Claude (paid)
LLM_PROVIDER=gemini          # or "anthropic"

# Vector Store: Switch between local and cloud
VECTOR_STORE_PROVIDER=chroma  # or "pinecone"
```

---

## 📊 Evaluation

The built-in evaluation pipeline tests 12 curated questions across 4 difficulty levels:

```bash
python notebooks/run_evaluation.py
```

| Metric | Score | Description |
|--------|-------|-------------|
| Retrieval Recall | 100% | Correct source found in top-5 results |
| Answer Correctness | 84.7% | Expected key facts present in answer |
| Faithfulness | ~70%* | Answer grounded in source text |

*Faithfulness uses a simple keyword overlap check. For production, integrate RAGAS or LLM-as-Judge.

---

## 📁 Project Structure

```
├── app.py                          # Streamlit frontend (glassmorphism UI)
├── src/
│   ├── api/main.py                 # FastAPI backend (14 endpoints)
│   ├── ingestion/
│   │   ├── parser.py               # Document parsing (PDF, DOCX, TXT)
│   │   └── chunker.py              # Recursive text chunking with overlap
│   ├── embeddings/
│   │   └── embedding_service.py    # Sentence-Transformers embeddings
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB with collections + filtering
│   │   ├── pinecone_store.py       # Pinecone cloud vector store
│   │   ├── store_provider.py       # Vector store factory (swappable)
│   │   └── hybrid_search.py        # BM25 + semantic + RRF fusion
│   ├── chains/
│   │   └── rag_chain.py            # RAG pipeline + memory + caching
│   ├── evaluation/
│   │   ├── test_dataset.py         # 12 curated Q&A test cases
│   │   └── evaluator.py            # Retrieval + correctness + faithfulness
│   ├── utils/
│   │   └── cache.py                # TTL-based response caching
│   └── llm_provider.py             # LLM factory (Gemini ↔ Claude)
├── configs/settings.py             # Pydantic environment config
├── tests/                          # 60+ automated tests
├── notebooks/                      # Demo and evaluation scripts
└── data/sample_docs/               # Sample legal documents
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_api.py -v          # API endpoint tests
pytest tests/test_vector_store.py -v # Vector store + embeddings
pytest tests/test_hybrid_search.py -v # BM25 + hybrid search
pytest tests/test_ingestion.py -v    # Parser + chunker
pytest tests/test_cache.py -v        # Caching module
```

---

## 🔮 What I'd Add Next

- **Streaming responses** — stream LLM output token-by-token for better UX
- **PDF table extraction** — handle structured tables in legal documents
- **LLM-as-Judge evaluation** — more accurate faithfulness scoring
- **Docker deployment** — containerized for cloud deployment
- **User authentication** — multi-tenant with separate document stores

---

## 📚 What I Learned

This project taught me the full RAG pipeline end-to-end:

- **Embeddings & vector search** — how text becomes searchable numbers
- **Hybrid retrieval** — why combining semantic + keyword search outperforms either alone
- **Prompt engineering** — constraining LLMs to only use provided context
- **Provider abstraction** — factory pattern for swappable backends
- **Evaluation methodology** — measuring RAG quality beyond "does it look right?"
- **Caching strategy** — when to cache, when to invalidate, TTL design

---

*Part of a 4-project AI/ML portfolio. See also: Smart City Data Agent · Code Instruction Fine-tuner · LLM Eval Benchmark.*
