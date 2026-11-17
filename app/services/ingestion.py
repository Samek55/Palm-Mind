import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import EmptyDocumentError
from app.db.models import Document
from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking import get_chunker
from app.services.embeddings import Embedder
from app.services.text_extraction import extract_text
from app.services.vector_store import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)

_CONTENT_TYPES = {".pdf": "application/pdf", ".txt": "text/plain"}


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(
        self,
        db: Session,
        filename: str,
        data: bytes,
        strategy: ChunkingStrategy,
    ) -> Document:
        # extract first so obviously bad files never create a DB row
        text = extract_text(filename, data)

        document = Document(
            filename=filename,
            content_type=_CONTENT_TYPES[Path(filename).suffix.lower()],
            size_bytes=len(data),
            chunking_strategy=strategy.value,
            char_count=len(text),
            embedding_model=self.settings.embedding_model,
            status="processing",
        )
        db.add(document)
        db.commit()

        try:
            chunker = get_chunker(
                strategy,
                self.embedder,
                chunk_size=self.settings.fixed_chunk_size,
                overlap=self.settings.fixed_chunk_overlap,
                similarity_threshold=self.settings.semantic_similarity_threshold,
                max_chunk_words=self.settings.semantic_max_chunk_words,
            )
            chunks = chunker.split(text)
            if not chunks:
                raise EmptyDocumentError("Document produced no chunks")

            vectors = self.embedder.embed(chunks)
            self.vector_store.ensure_collection(self.embedder.dimension)
            records = [
                ChunkRecord(
                    document_id=document.id,
                    filename=filename,
                    chunk_index=i,
                    text=chunk,
                    strategy=strategy.value,
                )
                for i, chunk in enumerate(chunks)
            ]
            self.vector_store.upsert(records, vectors)
        except Exception as exc:
            logger.exception("Ingestion failed for %s", filename)
            document.status = "failed"
            document.error = str(exc)[:1000]
            db.commit()
            raise

        document.chunk_count = len(chunks)
        document.status = "ready"
        db.commit()
        logger.info("Ingested %s (%d chunks, strategy=%s)", filename, len(chunks), strategy.value)
        return document
