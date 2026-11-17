from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.schemas.booking import BookingDraft
from app.services.booking import BookingService


def test_draft_merge_keeps_existing_values() -> None:
    draft = BookingDraft(name="Sita Rai", email="sita@example.com")
    merged = draft.merge(BookingDraft(email=None, date="2030-01-05"))

    assert merged.name == "Sita Rai"
    assert merged.email == "sita@example.com"
    assert merged.date == "2030-01-05"
    assert merged.missing_fields() == ["time"]


def test_try_create_valid_booking(db: Session) -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    draft = BookingDraft(name="Sita Rai", email="sita@example.com", date=tomorrow, time="14:30")

    attempt = BookingService().try_create(db, "session-1", draft)

    assert attempt.booking is not None
    assert attempt.booking.email == "sita@example.com"
    assert attempt.booking.interview_time.hour == 14


def test_try_create_reports_invalid_fields(db: Session) -> None:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    draft = BookingDraft(name="Sita Rai", email="not-an-email", date=yesterday, time="14:30")

    attempt = BookingService().try_create(db, "session-2", draft)

    assert attempt.booking is None
    assert set(attempt.invalid_fields) == {"email", "date"}


def test_list_bookings_filters_by_email(db: Session) -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    service = BookingService()
    service.try_create(db, "s", BookingDraft(name="A B", email="a@example.com", date=tomorrow, time="09:00"))
    service.try_create(db, "s", BookingDraft(name="C D", email="c@example.com", date=tomorrow, time="10:00"))

    results = service.list_bookings(db, email="c@example.com")
    assert [b.name for b in results] == ["C D"]
