import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    strategy: str


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None: ...

    def upsert(self, records: list[ChunkRecord], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]: ...


class QdrantVectorStore:
    def __init__(
        self,
        collection: str,
        url: str | None = None,
        api_key: str | None = None,
        local_path: str = "./data/qdrant",
    ) -> None:
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            Path(local_path).mkdir(parents=True, exist_ok=True)
            logger.info("QDRANT_URL not set, using embedded Qdrant at %s", local_path)
            self.client = QdrantClient(path=local_path)
        self.collection = collection

    def ensure_collection(self, dimension: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        )
        # payload index so filtering / deleting by document is fast
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="document_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    def upsert(self, records: list[ChunkRecord], vectors: list[list[float]]) -> None:
        if len(records) != len(vectors):
            raise ValueError("records and vectors must have the same length")

        points = [
            qm.PointStruct(
                # deterministic id so re-ingesting the same chunk overwrites it
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r.document_id}:{r.chunk_index}")),
                vector=vector,
                payload={
                    "document_id": r.document_id,
                    "filename": r.filename,
                    "chunk_index": r.chunk_index,
                    "text": r.text,
                    "strategy": r.strategy,
                },
            )
            for r, vector in zip(records, vectors, strict=True)
        ]
        # keep requests a reasonable size for big documents
        for i in range(0, len(points), 256):
            self.client.upsert(collection_name=self.collection, points=points[i : i + 256])

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if not self.client.collection_exists(self.collection):
            # nothing has been ingested yet
            return []

        query_filter = None
        if document_ids:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=document_ids))]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
                )
            ),
        )
