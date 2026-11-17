from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_booking_service, get_chat_memory, get_chat_service
from app.db.models import Booking
from app.db.session import get_db
from app.schemas.booking import BookingOut
from app.schemas.chat import ChatHistoryResponse, ChatRequest, ChatResponse
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.chat_memory import RedisChatMemory

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Send a message to the RAG assistant")
def chat(
    payload: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    return service.handle_message(db, payload.session_id, payload.message)


@router.get(
    "/chat/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get the stored conversation for a session",
)
def chat_history(
    session_id: str,
    memory: Annotated[RedisChatMemory, Depends(get_chat_memory)],
) -> ChatHistoryResponse:
    return ChatHistoryResponse(session_id=session_id, messages=memory.get_history(session_id))


@router.delete("/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Clear a session")
def clear_chat(
    session_id: str,
    memory: Annotated[RedisChatMemory, Depends(get_chat_memory)],
) -> None:
    memory.clear(session_id)


@router.get("/bookings", response_model=list[BookingOut], summary="List interview bookings")
def list_bookings(
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[BookingService, Depends(get_booking_service)],
    email: Annotated[str | None, Query(description="Filter by email")] = None,
) -> list[Booking]:
    return service.list_bookings(db, email)
