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
    description="RAG-powered Q&A system with collection-based document management.",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = DocumentParser()
chunker = TextChunker()
vector_store = VectorStore()
rag_chain = RAGChain()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Request Models ---

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    source_filters: list[str] | None = None


class AskRequest(BaseModel):
    question: str
    k: int = 5
    session_id: str = "default"
    source_filters: list[str] | None = None


# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "app": "Legal Document Q&A (RAG)",
        "version": "0.6.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "vector_store": "up",
            "llm": "up",
            "total_chunks_stored": stats["total_chunks"],
            "total_documents": stats["total_documents"],
            "total_collections": stats["total_collections"],
        },
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Query(..., description="Collection/folder name (required)"),
    force: bool = Query(default=False, description="Overwrite if document already exists"),
):
    """
    Upload a legal document into a specific collection.
    Collection is required — every document must belong to a folder.
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. "
                   f"Supported: {list(DocumentParser.SUPPORTED_EXTENSIONS)}",
        )

    if not collection.strip():
        raise HTTPException(status_code=400, detail="Collection name cannot be empty.")

    if not force and vector_store.document_exists(file.filename):
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already exists. "
                   f"Use force=true to overwrite.",
        )

    if force and vector_store.document_exists(file.filename):
        vector_store.delete_document(file.filename)

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
    num_stored = vector_store.add_chunks(
        chunks,
        doc_id=file.filename,
        collection_name=collection.strip(),
    )

    return {
        "filename": file.filename,
        "collection": collection.strip(),
        "pages_extracted": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": num_stored,
        "message": f"Stored '{file.filename}' in '{collection}' ({num_stored} chunks).",
    }


@app.get("/collections")
async def list_collections():
    """List all collections with their documents and chunk counts."""
    collections = vector_store.list_collections()
    return {
        "total_collections": len(collections),
        "collections": collections,
    }


@app.get("/documents")
async def list_documents():
    """List all uploaded documents (flat list with collection info)."""
    documents = vector_store.list_documents()
    return {
        "total_documents": len(documents),
        "documents": documents,
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a specific document and all its chunks."""
    if not vector_store.document_exists(filename):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    num_deleted = vector_store.delete_document(filename)
    file_path = UPLOAD_DIR / filename
    file_path.unlink(missing_ok=True)

    return {
        "message": f"Deleted '{filename}' ({num_deleted} chunks removed).",
        "chunks_deleted": num_deleted,
    }


@app.delete("/collections/{collection_name}")
async def delete_collection_group(collection_name: str):
    """Delete all documents in a collection."""
    num_deleted = vector_store.delete_collection_group(collection_name)
    if num_deleted == 0:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

    return {
        "message": f"Deleted collection '{collection_name}' ({num_deleted} chunks removed).",
        "chunks_deleted": num_deleted,
    }


@app.post("/search")
async def search_documents(request: SearchRequest):
    """Semantic search with optional multi-source filtering."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stats = vector_store.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")

    results = vector_store.search(
        query=request.query,
        k=request.k,
        source_filters=request.source_filters,
    )

    return {
        "query": request.query,
        "source_filters": request.source_filters,
        "num_results": len(results),
        "results": results,
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question with optional multi-source filtering."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    stats = vector_store.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")

    result = rag_chain.ask(
        question=request.question,
        k=request.k,
        session_id=request.session_id,
        source_filters=request.source_filters,
    )

    return result


@app.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    history = rag_chain.get_memory(session_id)
    return {
        "session_id": session_id,
        "num_messages": len(history),
        "messages": history,
    }


@app.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    rag_chain.clear_memory(session_id)
    return {"message": f"Conversation history cleared for session '{session_id}'."}


@app.get("/stats")
async def get_stats():
    return vector_store.get_stats()


@app.delete("/reset")
async def reset_database():
    vector_store.delete_collection()
    return {"message": "Vector store cleared. All documents deleted."}
