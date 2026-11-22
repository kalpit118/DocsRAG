"""
Unit and integration tests for the expanded DocsRAG features:
- Document upload, listing, raw viewing, and deletion
- Conversational session history and query reformulation
- SSE Streaming endpoint
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.schemas import Source


@pytest.fixture()
def client():
    with patch("src.api.main.retrieve") as mock_ret, \
         patch("src.api.main.rerank") as mock_rr, \
         patch("src.api.main.generate_stream") as mock_stream, \
         patch("src.api.main.score_faithfulness") as mock_judge:
        
        mock_ret.return_value = [
            Source(doc_id="docs/test.md", chunk_index=0, text="LangChain info", score=0.92)
        ]
        mock_rr.return_value = [
            Source(doc_id="docs/test.md", chunk_index=0, text="LangChain info", score=0.92)
        ]
        mock_stream.return_value = ["LangChain ", "is ", "great."]
        mock_judge.return_value = (True, 1.0)

        from src.api.main import app
        yield TestClient(app)


def test_upload_and_manage_documents(client):
    # 1. Upload a text document
    file_content = b"# Custom Agent Docs\n\nThis is a user uploaded custom document."
    files = {"files": ("custom_agent.md", io.BytesIO(file_content), "text/markdown")}

    with patch("src.api.main.ingest_user_document") as mock_ingest:
        mock_ingest.return_value = {
            "filename": "custom_agent.md",
            "size_bytes": len(file_content),
            "chunks_recursive": 1,
            "chunks_header": 1,
            "indexed_at": 123456789.0,
        }
        res = client.post("/documents/upload", files=files)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["total_files"] == 1
        assert data["uploaded"][0]["filename"] == "custom_agent.md"

    # 2. List documents
    with patch("src.api.main.list_user_documents") as mock_list:
        mock_list.return_value = [
            {"filename": "custom_agent.md", "size_bytes": 100, "chunks": 2, "modified_at": 1000.0}
        ]
        res = client.get("/documents")
        assert res.status_code == 200
        docs = res.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "custom_agent.md"

    # 3. Delete document
    with patch("src.api.main.delete_user_document") as mock_del:
        mock_del.return_value = True
        res = client.delete("/documents/custom_agent.md")
        assert res.status_code == 200
        assert res.json()["status"] == "deleted"


def test_raw_document_viewer_and_security(client):
    # 1. Valid raw document
    with patch("src.api.main.get_raw_document") as mock_raw:
        mock_raw.return_value = {
            "doc_id": "test.md",
            "filename": "test.md",
            "source_type": "user_upload",
            "content": "# Test Document Content",
            "size_bytes": 23,
        }
        res = client.get("/documents/raw?doc_id=test.md")
        assert res.status_code == 200
        assert res.json()["content"] == "# Test Document Content"

    # 2. Directory traversal attempt
    res = client.get("/documents/raw?doc_id=../../../../etc/passwd")
    assert res.status_code in (403, 404)


def test_session_history_endpoints(client):
    session_id = "test-session-123"

    # Fetch empty session
    res = client.get(f"/sessions/{session_id}/messages")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Delete session
    res = client.delete(f"/sessions/{session_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "cleared"


def test_ask_stream_sse(client):
    payload = {
        "question": "What is LangChain?",
        "chunking": "recursive",
        "use_reranker": False,
        "top_k": 3,
        "session_id": "sse-test-session",
    }
    with client.stream("POST", "/ask/stream", json=payload) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        text = "\n".join(lines)
        assert "event: metadata" in text
        assert "event: token" in text
        assert "event: done" in text
        assert "LangChain" in text
