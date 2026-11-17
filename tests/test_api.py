from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import Container
from app.core.config import get_settings
from app.main import create_app
from app.schemas.chat import Intent
from app.services.booking import BookingService
from app.services.chat import ChatService, RouterDecision
from app.services.ingestion import IngestionService


@pytest.fixture
def client(llm, embedder, vector_store, memory) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    settings = get_settings()
    bookings = BookingService()
    app = create_app()
    app.state.container = Container(
        ingestion=IngestionService(settings, embedder, vector_store),
        chat=ChatService(settings, llm, embedder, vector_store, memory, bookings),
        memory=memory,
        bookings=bookings,
    )
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("strategy", ["fixed", "semantic"])
def test_upload_txt(client: TestClient, vector_store, strategy: str) -> None:  # type: ignore[no-untyped-def]
    content = b"Palm Mind builds AI products. The team is based in Nepal. " * 30
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("about.txt", content, "text/plain")},
        data={"chunking_strategy": strategy},
    )

    assert resp.status_code == 201, resp.text
    doc = resp.json()["document"]
    assert doc["status"] == "ready"
    assert doc["chunking_strategy"] == strategy
    assert doc["chunk_count"] == len(vector_store.records) > 0

    fetched = client.get(f"/api/v1/documents/{doc['id']}")
    assert fetched.status_code == 200


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("x.docx", b"hello", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_upload_rejects_empty_file(client: TestClient) -> None:
    resp = client.post("/api/v1/documents/upload", files={"file": ("empty.txt", b"   ", "text/plain")})
    assert resp.status_code == 422


def test_upload_rejects_bad_strategy(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("a.txt", b"hello world", "text/plain")},
        data={"chunking_strategy": "magic"},
    )
    assert resp.status_code == 422


def test_chat_and_history(client: TestClient, llm) -> None:  # type: ignore[no-untyped-def]
    llm.structured = [RouterDecision(intent=Intent.CHITCHAT, standalone_query="hello")]

    resp = client.post("/api/v1/chat", json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "scripted answer"

    history = client.get(f"/api/v1/chat/{body['session_id']}/history").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
