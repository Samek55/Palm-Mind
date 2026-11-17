from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    char_count: int
    embedding_model: str
    status: str
    created_at: datetime


class IngestionResponse(BaseModel):
    document: DocumentOut
    message: str
