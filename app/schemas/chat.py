from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.booking import BookingOut


class Intent(str, Enum):
    QUESTION = "question"
    BOOKING = "booking"
    CHITCHAT = "chitchat"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime | None = None


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Omit on the first message; reuse the returned id for follow-ups.",
    )
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent
    sources: list[SourceChunk] = []
    booking: BookingOut | None = None
    pending_booking_fields: list[str] = []


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
