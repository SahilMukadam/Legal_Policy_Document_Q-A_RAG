"""
Legal/Policy Document Q&A — RAG Application
Main FastAPI entry point.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
            "vector_store": "not_initialized",  # Will update as we build
            "llm": "not_initialized",
        },
    }
