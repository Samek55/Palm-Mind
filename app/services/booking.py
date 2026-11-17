import logging
from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Booking
from app.schemas.booking import BookingCreate, BookingDraft

logger = logging.getLogger(__name__)

# BookingCreate field name -> BookingDraft field name
_FIELD_MAP = {"name": "name", "email": "email", "interview_date": "date", "interview_time": "time"}

FIELD_LABELS = {
    "name": "your full name",
    "email": "your email address",
    "date": "the preferred date",
    "time": "the preferred time",
}


@dataclass
class BookingAttempt:
    booking: Booking | None = None
    invalid_fields: dict[str, str] = field(default_factory=dict)  # draft field -> reason


class BookingService:
    def try_create(self, db: Session, session_id: str, draft: BookingDraft) -> BookingAttempt:
        """Validate a complete draft and persist it. Returns invalid fields instead of raising."""
        try:
            data = BookingCreate(
                name=draft.name or "",
                email=draft.email or "",
                interview_date=draft.date,  # pydantic parses the ISO string
                interview_time=draft.time,
            )
        except ValidationError as exc:
            invalid: dict[str, str] = {}
            for err in exc.errors():
                loc = str(err["loc"][0]) if err["loc"] else ""
                invalid[_FIELD_MAP.get(loc, loc)] = err["msg"]
            return BookingAttempt(invalid_fields=invalid)

        booking = Booking(
            session_id=session_id,
            name=data.name,
            email=str(data.email),
            interview_date=data.interview_date,
            interview_time=data.interview_time,
        )
        db.add(booking)
        db.commit()
        logger.info("Booking %s created for %s", booking.id, booking.email)
        return BookingAttempt(booking=booking)

    def list_bookings(self, db: Session, email: str | None = None) -> list[Booking]:
        stmt = select(Booking).order_by(Booking.created_at.desc())
        if email:
            stmt = stmt.where(Booking.email == email)
        return list(db.scalars(stmt))
