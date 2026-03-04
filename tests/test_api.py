"""
Smoke tests to verify project setup is working.
Run with: pytest tests/ -v
"""

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
