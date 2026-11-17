import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis import Redis

from app.api.deps import Container
from app.api.routes import chat, documents
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.db.session import init_db
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.chat_memory import RedisChatMemory
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.ingestion import IngestionService
from app.services.llm import LLMClient
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def build_container(settings: Settings) -> Container:
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vector_store = QdrantVectorStore(
        collection=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
    )
    memory = RedisChatMemory(
        Redis.from_url(settings.redis_url, decode_responses=True),
        max_messages=settings.chat_history_max_messages,
        ttl_seconds=settings.chat_session_ttl_seconds,
    )
    bookings = BookingService()
    llm = LLMClient(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    return Container(
        ingestion=IngestionService(settings, embedder, vector_store),
        chat=ChatService(settings, llm, embedder, vector_store, memory, bookings),
        memory=memory,
        bookings=bookings,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.debug)
    init_db()
    if not hasattr(app.state, "container"):  # tests may inject their own
        app.state.container = build_container(settings)
    logger.info("%s started", settings.app_name)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Document ingestion + conversational RAG with interview booking.",
        lifespan=lifespan,
    )

    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    return app


app = create_app()
