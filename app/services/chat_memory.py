import json
from datetime import UTC, datetime
from typing import Literal

from redis import Redis

from app.schemas.booking import BookingDraft
from app.schemas.chat import ChatMessage


class RedisChatMemory:
    """Stores conversation history and in-progress booking drafts per session.

    Keys:
        chat:{session_id}:messages  -> list of JSON-encoded ChatMessage
        chat:{session_id}:booking   -> hash with the partially filled booking
    """

    def __init__(self, client: Redis, max_messages: int = 20, ttl_seconds: int = 86400) -> None:
        self.redis = client
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"chat:{session_id}:messages"

    @staticmethod
    def _booking_key(session_id: str) -> str:
        return f"chat:{session_id}:booking"

    # ---- history -----------------------------------------------------------

    def add_message(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        message = ChatMessage(role=role, content=content, created_at=datetime.now(UTC))
        key = self._messages_key(session_id)
        pipe = self.redis.pipeline()
        pipe.rpush(key, message.model_dump_json())
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def get_history(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        start = -limit if limit else 0
        raw: list[bytes | str] = self.redis.lrange(self._messages_key(session_id), start, -1)  # type: ignore[assignment]
        return [ChatMessage.model_validate(json.loads(item)) for item in raw]

    def clear(self, session_id: str) -> None:
        self.redis.delete(self._messages_key(session_id), self._booking_key(session_id))

    # ---- booking draft -----------------------------------------------------

    def get_booking_draft(self, session_id: str) -> BookingDraft | None:
        data: dict[bytes | str, bytes | str] = self.redis.hgetall(self._booking_key(session_id))  # type: ignore[assignment]
        if not data:
            return None
        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }
        return BookingDraft.model_validate(decoded)

    def save_booking_draft(self, session_id: str, draft: BookingDraft) -> None:
        key = self._booking_key(session_id)
        values = {k: v for k, v in draft.model_dump().items() if v}
        pipe = self.redis.pipeline()
        pipe.delete(key)
        if values:
            pipe.hset(key, mapping=values)
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def clear_booking_draft(self, session_id: str) -> None:
        self.redis.delete(self._booking_key(session_id))
