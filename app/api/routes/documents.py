from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_ingestion_service
from app.core.config import Settings, get_settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileError
from app.db.models import Document
from app.db.session import get_db
from app.schemas.ingestion import ChunkingStrategy, DocumentOut, IngestionResponse
from app.services.ingestion import IngestionService
from app.services.text_extraction import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .pdf or .txt file and index it",
)
def upload_document(
    file: Annotated[UploadFile, File(description=".pdf or .txt file")],
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.FIXED,
) -> IngestionResponse:
    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise UnsupportedFileError("Only .pdf and .txt files are supported.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds the {settings.max_upload_size_mb} MB limit.")

    document = service.ingest(db, filename, data, chunking_strategy)
    return IngestionResponse(
        document=DocumentOut.model_validate(document),
        message=f"Indexed {document.chunk_count} chunks using '{chunking_strategy.value}' chunking.",
    )


@router.get("", response_model=list[DocumentOut], summary="List ingested documents")
def list_documents(db: Annotated[Session, Depends(get_db)]) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@router.get("/{document_id}", response_model=DocumentOut, summary="Get document metadata")
def get_document(document_id: str, db: Annotated[Session, Depends(get_db)]) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document
