"""
Legal/Policy Document Q&A — RAG Application
Main FastAPI entry point.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStore
from src.chains.rag_chain import RAGChain

app = FastAPI(
    title="Legal Document Q&A",
    description="RAG-powered Q&A system for legal and policy documents. "
                "Upload contracts, policies, or terms of service and ask questions "
                "to get cited answers from the source text. Supports multi-turn "
                "conversations with follow-up questions.",
    version="0.4.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
parser = DocumentParser()
chunker = TextChunker()
vector_store = VectorStore()
rag_chain = RAGChain()

# Upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Request/Response Models ---

class SearchRequest(BaseModel):
    """Request body for the search endpoint."""
    query: str
    k: int = 5


class AskRequest(BaseModel):
    """Request body for the ask endpoint."""
    question: str
    k: int = 5
    session_id: str = "default"


# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint — basic info."""
    return {
        "app": "Legal Document Q&A (RAG)",
        "version": "0.4.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "vector_store": "up",
            "llm": "up",
            "total_chunks_stored": stats["total_chunks"],
            "embedding_model": stats["embedding_model"],
        },
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document for processing.

    The document is parsed, chunked, embedded, and stored in the
    vector database for later retrieval.
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. "
                   f"Supported: {list(DocumentParser.SUPPORTED_EXTENSIONS)}",
        )

    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        documents = parser.parse(str(file_path))
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunker.chunk(documents)
    num_stored = vector_store.add_chunks(chunks, doc_id=file.filename)

    return {
        "filename": file.filename,
        "pages_extracted": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": num_stored,
        "message": f"Successfully processed and stored '{file.filename}' "
                   f"({num_stored} chunks in vector DB).",
    }


@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search uploaded documents using semantic search.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stats = vector_store.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents uploaded yet. Upload a document first via /upload.",
        )

    results = vector_store.search(query=request.query, k=request.k)

    return {
        "query": request.query,
        "num_results": len(results),
        "results": results,
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask a question about uploaded legal documents.

    Supports multi-turn conversations via session_id. Use the same
    session_id across requests to enable follow-up questions.

    Example first question:
        {"question": "When is rent due?", "session_id": "user1"}

    Example follow-up:
        {"question": "What if I pay late?", "session_id": "user1"}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    stats = vector_store.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents uploaded yet. Upload a document first via /upload.",
        )

    result = rag_chain.ask(
        question=request.question,
        k=request.k,
        session_id=request.session_id,
    )

    return result


@app.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a session.
    Useful for displaying chat history in a frontend.
    """
    history = rag_chain.get_memory(session_id)
    return {
        "session_id": session_id,
        "num_messages": len(history),
        "messages": history,
    }


@app.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session."""
    rag_chain.clear_memory(session_id)
    return {
        "message": f"Conversation history cleared for session '{session_id}'.",
    }


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    return vector_store.get_stats()


@app.delete("/reset")
async def reset_database():
    """Delete all stored documents. Use with caution."""
    vector_store.delete_collection()
    return {"message": "Vector store cleared. All documents deleted."}
