from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.schemas.booking import BookingDraft
from app.schemas.chat import Intent
from app.services.booking import BookingService
from app.services.chat import ChatService, RouterDecision
from app.services.vector_store import ChunkRecord


@pytest.fixture
def chat_service(llm, embedder, vector_store, memory) -> ChatService:  # type: ignore[no-untyped-def]
    settings = Settings(retrieval_score_threshold=0.0)
    return ChatService(settings, llm, embedder, vector_store, memory, BookingService())


def test_multi_turn_booking_flow(chat_service: ChatService, llm, memory, db: Session) -> None:  # type: ignore[no-untyped-def]
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    llm.structured = [
        RouterDecision(intent=Intent.BOOKING, standalone_query="book interview"),
        RouterDecision(
            intent=Intent.BOOKING,
            standalone_query="my details",
            booking=BookingDraft(name="Ram Sharma", email="ram@example.com"),
        ),
        RouterDecision(
            intent=Intent.BOOKING,
            standalone_query="tomorrow 3pm",
            booking=BookingDraft(date=tomorrow, time="15:00"),
        ),
    ]

    first = chat_service.handle_message(db, None, "I'd like to book an interview")
    sid = first.session_id
    assert first.pending_booking_fields == ["name", "email", "date", "time"]

    second = chat_service.handle_message(db, sid, "I'm Ram Sharma, ram@example.com")
    assert second.pending_booking_fields == ["date", "time"]
    assert "Ram" in second.answer

    third = chat_service.handle_message(db, sid, "tomorrow at 3pm")
    assert third.booking is not None
    assert third.booking.email == "ram@example.com"
    assert memory.get_booking_draft(sid) is None
    assert len(memory.get_history(sid)) == 6


def test_invalid_email_is_asked_again(chat_service: ChatService, llm, memory, db: Session) -> None:  # type: ignore[no-untyped-def]
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    llm.structured = [
        RouterDecision(
            intent=Intent.BOOKING,
            standalone_query="book",
            booking=BookingDraft(name="Ram Sharma", email="ram-at-example", date=tomorrow, time="10:00"),
        )
    ]

    response = chat_service.handle_message(db, "s-invalid", "book me in")

    assert response.booking is None
    assert response.pending_booking_fields == ["email"]
    draft = memory.get_booking_draft("s-invalid")
    assert draft is not None and draft.name == "Ram Sharma"


def test_cancel_booking_clears_draft(chat_service: ChatService, llm, memory, db: Session) -> None:  # type: ignore[no-untyped-def]
    memory.save_booking_draft("s-cancel", BookingDraft(name="Ram Sharma"))
    llm.structured = [RouterDecision(intent=Intent.BOOKING, standalone_query="cancel", cancel_booking=True)]

    response = chat_service.handle_message(db, "s-cancel", "never mind, cancel it")

    assert "cancelled" in response.answer
    assert memory.get_booking_draft("s-cancel") is None


def test_question_uses_retrieved_context_and_history(
    chat_service: ChatService,
    llm,
    embedder,
    vector_store,
    db: Session,  # type: ignore[no-untyped-def]
) -> None:
    chunks = ["The Pro plan costs 50 dollars per month.", "Our office is located in Kathmandu."]
    vector_store.upsert(
        [ChunkRecord("doc-1", "pricing.txt", i, c, "fixed") for i, c in enumerate(chunks)],
        embedder.embed(chunks),
    )
    llm.structured = [
        RouterDecision(intent=Intent.CHITCHAT, standalone_query="hi"),
        RouterDecision(intent=Intent.QUESTION, standalone_query="How much does the Pro plan cost per month?"),
    ]

    chat_service.handle_message(db, "s-q", "hi")
    response = chat_service.handle_message(db, "s-q", "how much is it?")

    assert response.intent is Intent.QUESTION
    assert response.sources[0].text == chunks[0]
    sent = llm.text_calls[-1]
    # previous turn is included, and the final user turn carries the context
    assert sent[0] == {"role": "user", "content": "hi"}
    assert "Pro plan costs 50 dollars" in sent[-1]["content"]
