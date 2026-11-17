import logging
import threading
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local embedding model. Loaded lazily because it takes a few seconds."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # imported here because it pulls in torch, which is slow to load
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding size for {self.model_name}")
        return dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
