from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator


class BookingDraft(BaseModel):
    """Partially collected booking details, kept in Redis between turns.

    Everything is a plain optional string here because it comes straight from
    the LLM; real validation happens when we convert it to BookingCreate.
    """

    name: str | None = None
    email: str | None = None
    date: str | None = None  # YYYY-MM-DD
    time: str | None = None  # HH:MM (24h)

    def merge(self, other: "BookingDraft") -> "BookingDraft":
        updates = {k: v for k, v in other.model_dump().items() if v not in (None, "")}
        return self.model_copy(update=updates)

    def missing_fields(self) -> list[str]:
        return [field for field, value in self.model_dump().items() if not value]


class BookingCreate(BaseModel):
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("name is too short")
        return value

    @field_validator("interview_date")
    @classmethod
    def _not_in_past(cls, value: date) -> date:
        if value < date.today():
            raise ValueError("date cannot be in the past")
        return value


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    status: str
    created_at: datetime
