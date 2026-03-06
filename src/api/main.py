"""
Legal/Policy Document Q&A — RAG Application
Main FastAPI entry point.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker

app = FastAPI(
    title="Legal Document Q&A",
    description="RAG-powered Q&A system for legal and policy documents. "
                "Upload contracts, policies, or terms of service and ask questions "
                "to get cited answers from the source text.",
    version="0.1.0",
)

# CORS — allow all origins in development
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

# Upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint — basic info."""
    return {
        "app": "Legal Document Q&A (RAG)",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "vector_store": "not_initialized",
            "llm": "not_initialized",
        },
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document for processing.

    Accepts PDF, DOCX, or TXT files. The document is:
    1. Saved to the upload directory
    2. Parsed to extract text
    3. Chunked into smaller passages for retrieval

    Returns the number of chunks created and a preview.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. "
                   f"Supported: {list(DocumentParser.SUPPORTED_EXTENSIONS)}",
        )

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse the document
    try:
        documents = parser.parse(str(file_path))
    except ValueError as e:
        # Clean up the saved file if parsing fails
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    # Chunk the documents
    chunks = chunker.chunk(documents)

    return {
        "filename": file.filename,
        "pages_extracted": len(documents),
        "chunks_created": len(chunks),
        "chunk_preview": [
            {
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "metadata": chunk["metadata"],
            }
            for chunk in chunks[:3]  # Show first 3 chunks as preview
        ],
        "message": f"Successfully processed '{file.filename}' into {len(chunks)} chunks.",
    }
