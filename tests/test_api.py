"""
API endpoint tests.
Run with: pytest tests/test_api.py -v
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
        "and use your personal information. We do not sell your data. "
        "All data is encrypted at rest and in transit.\n\n"
        "DATA RETENTION\n\n"
        "We retain your data for a maximum of 3 years after account closure. "
        "You may request deletion at any time by contacting support."
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
    assert data["chunks_stored"] >= 1


def test_upload_unsupported_file():
    """Test that unsupported file types are rejected."""
    file = io.BytesIO(b"some data")

    response = client.post(
        "/upload",
        files={"file": ("data.xlsx", file, "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_search_after_upload():
    """Test searching after uploading a document."""
    # First upload a document
    content = (
        "LEASE AGREEMENT\n\n"
        "The tenant shall pay monthly rent of 1500 pounds by the first "
        "of each month. Late payments incur a 5 percent penalty.\n\n"
        "The landlord is responsible for structural repairs and maintenance "
        "of common areas. The tenant must keep the unit clean and tidy."
    )
    file = io.BytesIO(content.encode("utf-8"))
    client.post(
        "/upload",
        files={"file": ("lease.txt", file, "text/plain")},
    )

    # Now search
    response = client.post(
        "/search",
        json={"query": "When is rent due?", "k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "When is rent due?"
    assert data["num_results"] >= 1
    assert len(data["results"]) >= 1
    # Each result should have text, metadata, and score
    assert "text" in data["results"][0]
    assert "metadata" in data["results"][0]
    assert "score" in data["results"][0]


def test_search_empty_query():
    """Test that empty queries are rejected."""
    response = client.post(
        "/search",
        json={"query": "", "k": 5},
    )
    assert response.status_code == 400


def test_stats_endpoint():
    """Test the stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "embedding_model" in data
