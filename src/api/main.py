"""
Legal/Policy Document Q&A — RAG Application
Main FastAPI entry point.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
                "conversations, document filtering, and multi-document management.",
    version="0.5.0",
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
    source_filter: str | None = None


class AskRequest(BaseModel):
    """Request body for the ask endpoint."""
    question: str
    k: int = 5
    session_id: str = "default"
    source_filter: str | None = None


# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint — basic info."""
    return {
        "app": "Legal Document Q&A (RAG)",
        "version": "0.5.0",
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
            "total_documents": stats["total_documents"],
            "embedding_model": stats["embedding_model"],
        },
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Query(default=False, description="Overwrite if document already exists"),
):
    """
    Upload a legal document for processing.

    The document is parsed, chunked, embedded, and stored in the
    vector database. Set force=true to re-upload an existing document.
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. "
                   f"Supported: {list(DocumentParser.SUPPORTED_EXTENSIONS)}",
        )

    # Check for duplicates
    if not force and vector_store.document_exists(file.filename):
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already exists. "
                   f"Use force=true to overwrite, or delete it first.",
        )

    # If forcing re-upload, delete existing chunks first
    if force and vector_store.document_exists(file.filename):
        vector_store.delete_document(file.filename)

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse
    try:
        documents = parser.parse(str(file_path))
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    # Chunk and store
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


@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents with their chunk counts.
    Useful for knowing what's available to search/ask about.
    """
    documents = vector_store.list_documents()
    return {
        "total_documents": len(documents),
        "documents": documents,
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a specific document and all its chunks from the vector store.
    """
    if not vector_store.document_exists(filename):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found.",
        )

    num_deleted = vector_store.delete_document(filename)

    # Also remove the file from uploads
    file_path = UPLOAD_DIR / filename
    file_path.unlink(missing_ok=True)

    return {
        "message": f"Deleted '{filename}' ({num_deleted} chunks removed).",
        "chunks_deleted": num_deleted,
    }


@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search uploaded documents using semantic search.
    Optionally filter by source document.

    Example: {"query": "rent payments", "source_filter": "lease.pdf"}
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stats = vector_store.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents uploaded yet. Upload a document first via /upload.",
        )

    results = vector_store.search(
        query=request.query,
        k=request.k,
        source_filter=request.source_filter,
    )

    return {
        "query": request.query,
        "source_filter": request.source_filter,
        "num_results": len(results),
        "results": results,
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask a question about uploaded legal documents.

    Supports multi-turn conversations via session_id and
    optional source_filter to restrict to specific documents.

    Example:
        {"question": "When is rent due?", "session_id": "user1", "source_filter": "lease.pdf"}
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
        source_filter=request.source_filter,
    )

    return result


@app.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
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
    """Get detailed vector store statistics."""
    return vector_store.get_stats()


@app.delete("/reset")
async def reset_database():
    """Delete all stored documents and chunks. Use with caution."""
    vector_store.delete_collection()
    return {"message": "Vector store cleared. All documents deleted."}
