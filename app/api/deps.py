from dataclasses import dataclass

from fastapi import Request

from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.chat_memory import RedisChatMemory
from app.services.ingestion import IngestionService


@dataclass
class Container:
    """Long-lived services, built once in the app lifespan."""

    ingestion: IngestionService
    chat: ChatService
    memory: RedisChatMemory
    bookings: BookingService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_ingestion_service(request: Request) -> IngestionService:
    return get_container(request).ingestion


def get_chat_service(request: Request) -> ChatService:
    return get_container(request).chat


def get_chat_memory(request: Request) -> RedisChatMemory:
    return get_container(request).memory


def get_booking_service(request: Request) -> BookingService:
    return get_container(request).bookings
