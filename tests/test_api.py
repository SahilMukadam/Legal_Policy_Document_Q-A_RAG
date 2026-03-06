"""
API endpoint tests.
Run with: pytest tests/ -v
"""

import io

from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    """Verify the root endpoint returns app info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "Legal Document Q&A (RAG)"
    assert data["status"] == "running"


def test_health_check():
    """Verify the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data


def test_upload_txt_file():
    """Test uploading a .txt document."""
    content = (
        "PRIVACY POLICY\n\n"
        "We respect your privacy. This policy explains how we collect "
        "and use your personal information. We do not sell your data."
    )
    file = io.BytesIO(content.encode("utf-8"))

    response = client.post(
        "/upload",
        files={"file": ("privacy_policy.txt", file, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "privacy_policy.txt"
    assert data["pages_extracted"] == 1
    assert data["chunks_created"] >= 1
    assert "chunk_preview" in data


def test_upload_unsupported_file():
    """Test that unsupported file types are rejected."""
    file = io.BytesIO(b"some data")

    response = client.post(
        "/upload",
        files={"file": ("data.xlsx", file, "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
