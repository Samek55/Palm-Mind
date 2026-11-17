import hashlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

# Point the app at a throwaway database before anything imports the settings.
_tmp_dir = tempfile.mkdtemp(prefix="rag-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp_dir, 'test.db').as_posix()}"
os.environ.setdefault("GROQ_API_KEY", "test-key")

import fakeredis  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.services.chat_memory import RedisChatMemory  # noqa: E402
from app.services.vector_store import ChunkRecord, SearchHit  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    dimension = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = np.zeros(self.dimension)
            for word in text.lower().split():
                word = word.strip(".,!?")
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec) or 1.0
            out.append((vec / norm).tolist())
        return out


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: list[tuple[ChunkRecord, list[float]]] = []
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def upsert(self, records: list[ChunkRecord], vectors: list[list[float]]) -> None:
        self.records.extend(zip(records, vectors, strict=True))

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        scored = [
            SearchHit(r.document_id, r.filename, r.chunk_index, r.text, float(np.dot(vector, v)))
            for r, v in self.records
        ]
        if score_threshold is not None:
            scored = [h for h in scored if h.score >= score_threshold]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]


class ScriptedLLM:
    """Returns pre-baked structured outputs in order, and a fixed text answer."""

    def __init__(self) -> None:
        self.structured: list[BaseModel] = []
        self.text_calls: list[list[dict[str, str]]] = []

    def generate_structured(self, system, messages, output_model):  # type: ignore[no-untyped-def]
        assert self.structured, "no scripted structured response left"
        return self.structured.pop(0)

    def generate_text(self, system, messages):  # type: ignore[no-untyped-def]
        self.text_calls.append(messages)
        return "scripted answer"


@pytest.fixture(scope="session", autouse=True)
def _create_tables() -> None:
    init_db()


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def memory() -> RedisChatMemory:
    return RedisChatMemory(fakeredis.FakeRedis(decode_responses=True), max_messages=20, ttl_seconds=60)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()
