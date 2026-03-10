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


def test_upload_txt_file():
    """Test uploading a .txt document."""
    content = (
        "PRIVACY POLICY\n\n"
        "We respect your privacy. This policy explains how we collect "
        "and use your personal information. We do not sell your data."
    )
    file = io.BytesIO(content.encode("utf-8"))

    response = client.post(
        "/upload?force=true",
        files={"file": ("privacy_policy.txt", file, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "privacy_policy.txt"
    assert data["chunks_stored"] >= 1


def test_upload_duplicate_blocked():
    """Test that re-uploading same file is blocked without force."""
    content = "DUPLICATE TEST DOCUMENT\n\nThis is a test."
    file = io.BytesIO(content.encode("utf-8"))
    client.post("/upload", files={"file": ("dup_test.txt", file, "text/plain")})

    # Try uploading again — should be blocked
    file2 = io.BytesIO(content.encode("utf-8"))
    response = client.post("/upload", files={"file": ("dup_test.txt", file2, "text/plain")})
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_upload_duplicate_with_force():
    """Test that force=true allows re-upload."""
    content = "FORCE UPLOAD TEST\n\nUpdated content."
    file = io.BytesIO(content.encode("utf-8"))
    response = client.post(
        "/upload?force=true",
        files={"file": ("dup_test.txt", file, "text/plain")},
    )
    assert response.status_code == 200


def test_upload_unsupported_file():
    """Test that unsupported file types are rejected."""
    file = io.BytesIO(b"some data")
    response = client.post(
        "/upload",
        files={"file": ("data.xlsx", file, "application/octet-stream")},
    )
    assert response.status_code == 400


def test_list_documents():
    """Test listing uploaded documents."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_search_with_source_filter():
    """Test filtered search within a specific document."""
    # Upload two different documents
    content1 = "LEASE TERMS\n\nRent is 1500 pounds per month due on the first."
    content2 = "EMPLOYMENT CONTRACT\n\nSalary is 50000 pounds per year paid monthly."
    file1 = io.BytesIO(content1.encode("utf-8"))
    file2 = io.BytesIO(content2.encode("utf-8"))
    client.post("/upload?force=true", files={"file": ("lease_doc.txt", file1, "text/plain")})
    client.post("/upload?force=true", files={"file": ("employment_doc.txt", file2, "text/plain")})

    # Search only within lease document
    response = client.post(
        "/search",
        json={"query": "How much is the payment?", "k": 3, "source_filter": "lease_doc.txt"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_filter"] == "lease_doc.txt"
    # All results should be from the lease document
    for result in data["results"]:
        assert result["metadata"]["source"] == "lease_doc.txt"


def test_search_empty_query():
    """Test that empty queries are rejected."""
    response = client.post("/search", json={"query": "", "k": 5})
    assert response.status_code == 400


def test_ask_empty_question():
    """Test that empty questions are rejected."""
    response = client.post("/ask", json={"question": "", "k": 5})
    assert response.status_code == 400


def test_ask_with_session_id():
    """Test that /ask accepts a session_id parameter."""
    content = "CONTRACT TERMS\n\nThe payment term is net 30 days from invoice date."
    file = io.BytesIO(content.encode("utf-8"))
    client.post("/upload?force=true", files={"file": ("contract.txt", file, "text/plain")})

    response = client.post(
        "/ask",
        json={"question": "What is the payment term?", "k": 3, "session_id": "test_session"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session"
    assert "answer" in data


def test_conversation_history_endpoint():
    """Test getting conversation history."""
    response = client.get("/history/some_session")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "some_session"
    assert "messages" in data


def test_clear_conversation_history():
    """Test clearing conversation history."""
    response = client.delete("/history/some_session")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"]


def test_delete_document():
    """Test deleting a specific document."""
    # Upload a document to delete
    content = "DELETE ME\n\nThis document will be deleted."
    file = io.BytesIO(content.encode("utf-8"))
    client.post("/upload?force=true", files={"file": ("to_delete.txt", file, "text/plain")})

    # Delete it
    response = client.delete("/documents/to_delete.txt")
    assert response.status_code == 200
    assert response.json()["chunks_deleted"] >= 1

    # Verify it's gone
    response = client.delete("/documents/to_delete.txt")
    assert response.status_code == 404


def test_stats_endpoint():
    """Test the stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "total_documents" in data
    assert "documents" in data
