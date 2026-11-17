import logging
import uuid
from datetime import date

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.schemas.booking import BookingDraft, BookingOut
from app.schemas.chat import ChatMessage, ChatResponse, Intent, SourceChunk
from app.services.booking import FIELD_LABELS, BookingService
from app.services.chat_memory import RedisChatMemory
from app.services.embeddings import Embedder
from app.services.llm import LLMClient, LLMMessage
from app.services.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CHITCHAT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from app.services.vector_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

# how many previous messages we show the LLM
_HISTORY_WINDOW = 10


class RouterDecision(BaseModel):
    intent: Intent
    standalone_query: str
    booking: BookingDraft = Field(default_factory=BookingDraft)
    cancel_booking: bool = False


class ChatService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedder: Embedder,
        vector_store: VectorStore,
        memory: RedisChatMemory,
        booking_service: BookingService,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.memory = memory
        self.bookings = booking_service

    def handle_message(self, db: Session, session_id: str | None, message: str) -> ChatResponse:
        session_id = session_id or uuid.uuid4().hex
        history = self.memory.get_history(session_id, limit=_HISTORY_WINDOW)
        draft = self.memory.get_booking_draft(session_id)

        decision = self._route(history, message, draft)
        logger.debug("session=%s decision=%s", session_id, decision.model_dump())

        if decision.cancel_booking and draft:
            self.memory.clear_booking_draft(session_id)
            response = ChatResponse(
                session_id=session_id,
                intent=Intent.BOOKING,
                answer="No problem, I've cancelled the booking request. Anything else I can help with?",
            )
        elif decision.intent is Intent.BOOKING or (draft and _has_any_field(decision.booking)):
            response = self._handle_booking(db, session_id, draft, decision.booking)
        elif decision.intent is Intent.QUESTION:
            response = self._answer_question(session_id, history, message, decision.standalone_query, draft)
        else:
            response = self._chitchat(session_id, history, message)

        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response.answer)
        return response

    # ---- routing -----------------------------------------------------------

    def _route(self, history: list[ChatMessage], message: str, draft: BookingDraft | None) -> RouterDecision:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in history) or "(no previous messages)"
        today = date.today()
        booking_state = (
            f"A booking is in progress. Collected so far: {draft.model_dump(exclude_none=True)}. "
            f"Still missing: {draft.missing_fields()}."
            if draft
            else "No booking in progress."
        )
        prompt = (
            f"Today is {today.isoformat()} ({today.strftime('%A')}).\n"
            f"{booking_state}\n\n"
            f"<conversation>\n{transcript}\n</conversation>\n\n"
            f"<latest_message>\n{message}\n</latest_message>"
        )
        return self.llm.generate_structured(
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_model=RouterDecision,
        )

    # ---- booking -----------------------------------------------------------

    def _handle_booking(
        self,
        db: Session,
        session_id: str,
        draft: BookingDraft | None,
        extracted: BookingDraft,
    ) -> ChatResponse:
        draft = (draft or BookingDraft()).merge(extracted)
        missing = draft.missing_fields()

        if missing:
            self.memory.save_booking_draft(session_id, draft)
            return ChatResponse(
                session_id=session_id,
                intent=Intent.BOOKING,
                answer=_ask_for(missing, draft),
                pending_booking_fields=missing,
            )

        attempt = self.bookings.try_create(db, session_id, draft)
        if attempt.booking is None:
            # drop the bad values so we ask for them again
            cleaned = draft.model_copy(update={f: None for f in attempt.invalid_fields})
            self.memory.save_booking_draft(session_id, cleaned)
            problems = "; ".join(
                f"{FIELD_LABELS.get(f, f)}: {reason}" for f, reason in attempt.invalid_fields.items()
            )
            return ChatResponse(
                session_id=session_id,
                intent=Intent.BOOKING,
                answer=f"Some details need fixing ({problems}). Could you send them again?",
                pending_booking_fields=cleaned.missing_fields(),
            )

        self.memory.clear_booking_draft(session_id)
        booking = attempt.booking
        return ChatResponse(
            session_id=session_id,
            intent=Intent.BOOKING,
            answer=(
                f"Your interview is booked, {booking.name}! "
                f"{booking.interview_date:%A, %d %B %Y} at {booking.interview_time:%H:%M}. "
                f"A confirmation will be sent to {booking.email}."
            ),
            booking=BookingOut.model_validate(booking),
        )

    # ---- RAG ---------------------------------------------------------------

    def _retrieve(self, query: str) -> list[SearchHit]:
        [vector] = self.embedder.embed([query])
        return self.vector_store.search(
            vector,
            top_k=self.settings.retrieval_top_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )

    def _answer_question(
        self,
        session_id: str,
        history: list[ChatMessage],
        message: str,
        standalone_query: str,
        draft: BookingDraft | None,
    ) -> ChatResponse:
        hits = self._retrieve(standalone_query)

        if hits:
            context = "\n\n".join(
                f'<excerpt source="{h.filename}" chunk="{h.chunk_index}">\n{h.text}\n</excerpt>' for h in hits
            )
        else:
            context = "(no relevant excerpts were found in the uploaded documents)"

        user_turn = f"<context>\n{context}\n</context>\n\nQuestion: {message}"
        if draft:
            user_turn += (
                "\n\n(Note: the user has an unfinished interview booking, still missing "
                f"{', '.join(draft.missing_fields())}. After answering, briefly remind them.)"
            )

        answer = self.llm.generate_text(
            system=ANSWER_SYSTEM_PROMPT,
            messages=[*_to_llm_messages(history), {"role": "user", "content": user_turn}],
        )
        return ChatResponse(
            session_id=session_id,
            intent=Intent.QUESTION,
            answer=answer,
            sources=[
                SourceChunk(
                    document_id=h.document_id,
                    filename=h.filename,
                    chunk_index=h.chunk_index,
                    score=round(h.score, 4),
                    text=h.text,
                )
                for h in hits
            ],
        )

    def _chitchat(self, session_id: str, history: list[ChatMessage], message: str) -> ChatResponse:
        answer = self.llm.generate_text(
            system=CHITCHAT_SYSTEM_PROMPT,
            messages=[*_to_llm_messages(history), {"role": "user", "content": message}],
        )
        return ChatResponse(session_id=session_id, intent=Intent.CHITCHAT, answer=answer)


def _has_any_field(draft: BookingDraft) -> bool:
    return any(draft.model_dump().values())


def _ask_for(missing: list[str], draft: BookingDraft) -> str:
    labels = [FIELD_LABELS[f] for f in missing]
    needed = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"
    greeting = f"Thanks {draft.name.split()[0]}! " if draft.name else ""
    return f"{greeting}To book your interview I still need {needed}."


def _to_llm_message(message: ChatMessage) -> LLMMessage:
    if message.role == "user":
        return {"role": "user", "content": message.content}
    return {"role": "assistant", "content": message.content}


def _to_llm_messages(history: list[ChatMessage]) -> list[LLMMessage]:
    """Convert stored history to API messages. The API expects the first one to be from the user."""
    messages = [_to_llm_message(m) for m in history]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages
