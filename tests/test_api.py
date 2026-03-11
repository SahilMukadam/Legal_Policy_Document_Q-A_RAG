"""
API endpoint tests with collection support.
Run with: pytest tests/test_api.py -v
"""

import io

from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_requires_collection():
    """Upload without collection should fail."""
    content = "TEST DOC\n\nSome content."
    file = io.BytesIO(content.encode("utf-8"))
    response = client.post(
        "/upload",
        files={"file": ("test.txt", file, "text/plain")},
    )
    # Missing required query param 'collection'
    assert response.status_code == 422


def test_upload_with_collection():
    """Upload with collection should succeed."""
    content = "LEASE AGREEMENT\n\nRent is 1500 per month due on the first."
    file = io.BytesIO(content.encode("utf-8"))
    response = client.post(
        "/upload?collection=Real+Estate&force=true",
        files={"file": ("lease.txt", file, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["collection"] == "Real Estate"
    assert data["chunks_stored"] >= 1


def test_upload_second_collection():
    """Upload to a different collection."""
    content = "EMPLOYMENT OFFER\n\nSalary is 50000 per year."
    file = io.BytesIO(content.encode("utf-8"))
    response = client.post(
        "/upload?collection=Employment&force=true",
        files={"file": ("offer.txt", file, "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["collection"] == "Employment"


def test_upload_duplicate_blocked():
    """Re-uploading same file without force should be blocked."""
    content = "DUPLICATE DOC\n\nTest content."
    file = io.BytesIO(content.encode("utf-8"))
    client.post(
        "/upload?collection=Test&force=true",
        files={"file": ("dup.txt", file, "text/plain")},
    )

    file2 = io.BytesIO(content.encode("utf-8"))
    response = client.post(
        "/upload?collection=Test",
        files={"file": ("dup.txt", file2, "text/plain")},
    )
    assert response.status_code == 409


def test_upload_unsupported_file():
    file = io.BytesIO(b"data")
    response = client.post(
        "/upload?collection=Test",
        files={"file": ("data.xlsx", file, "application/octet-stream")},
    )
    assert response.status_code == 400


def test_list_collections():
    """Collections endpoint should show organized structure."""
    response = client.get("/collections")
    assert response.status_code == 200
    data = response.json()
    assert "total_collections" in data
    assert "collections" in data


def test_list_documents():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    # Documents should have collection field
    if data["documents"]:
        assert "collection" in data["documents"][0]


def test_search_with_source_filters():
    """Search filtered to specific files."""
    response = client.post(
        "/search",
        json={"query": "payment", "k": 3, "source_filters": ["lease.txt"]},
    )
    assert response.status_code == 200
    data = response.json()
    for result in data["results"]:
        assert result["metadata"]["source"] == "lease.txt"


def test_search_empty_query():
    response = client.post("/search", json={"query": "", "k": 5})
    assert response.status_code == 400


def test_ask_with_session():
    response = client.post(
        "/ask",
        json={"question": "What is the rent?", "k": 3, "session_id": "test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test"
    assert "answer" in data


def test_ask_empty_question():
    response = client.post("/ask", json={"question": "", "k": 5})
    assert response.status_code == 400


def test_conversation_history():
    response = client.get("/history/test_session")
    assert response.status_code == 200
    assert "messages" in response.json()


def test_clear_history():
    response = client.delete("/history/test_session")
    assert response.status_code == 200


def test_delete_document():
    content = "TO DELETE\n\nThis will be deleted."
    file = io.BytesIO(content.encode("utf-8"))
    client.post(
        "/upload?collection=Temp&force=true",
        files={"file": ("deleteme.txt", file, "text/plain")},
    )

    response = client.delete("/documents/deleteme.txt")
    assert response.status_code == 200
    assert response.json()["chunks_deleted"] >= 1


def test_delete_collection_group():
    """Delete entire collection."""
    content = "TEMP DOC\n\nTemp content for collection deletion test."
    file = io.BytesIO(content.encode("utf-8"))
    client.post(
        "/upload?collection=ToDelete&force=true",
        files={"file": ("temp_coll.txt", file, "text/plain")},
    )

    response = client.delete("/collections/ToDelete")
    assert response.status_code == 200
    assert response.json()["chunks_deleted"] >= 1


def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "total_collections" in data
    assert "collections" in data
