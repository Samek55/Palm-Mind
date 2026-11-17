"""Chunking strategies.

fixed    - sliding window over words with overlap. Predictable chunk sizes, cheap.
semantic - split into sentences, embed them, and start a new chunk whenever the
           similarity between neighbouring sentences drops (i.e. topic shift).
"""

import re
from typing import Protocol

import numpy as np

from app.schemas.ingestion import ChunkingStrategy
from app.services.embeddings import Embedder

# Split after ., ! or ? followed by whitespace, or on blank lines.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s and s.strip()]


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(words), step):
            chunks.append(" ".join(words[start : start + self.chunk_size]))
            if start + self.chunk_size >= len(words):
                break
        return chunks


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.55,
        max_chunk_words: int = 300,
        min_chunk_words: int = 40,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_words = max_chunk_words
        # without a floor, short documents (letters, forms) end up as one-line chunks
        self.min_chunk_words = min_chunk_words

    def split(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return sentences

        # embeddings are L2-normalised, so the dot product is cosine similarity
        vectors = np.asarray(self.embedder.embed(sentences))
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_words = len(sentences[0].split())

        for sentence, sim in zip(sentences[1:], similarities, strict=True):
            n_words = len(sentence.split())
            topic_shift = sim < self.similarity_threshold and current_words >= self.min_chunk_words
            too_long = current_words + n_words > self.max_chunk_words
            if topic_shift or too_long:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            current.append(sentence)
            current_words += n_words

        if current:
            chunks.append(" ".join(current))
        return chunks


def get_chunker(
    strategy: ChunkingStrategy,
    embedder: Embedder,
    *,
    chunk_size: int,
    overlap: int,
    similarity_threshold: float,
    max_chunk_words: int,
    min_chunk_words: int,
) -> Chunker:
    if strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy is ChunkingStrategy.SEMANTIC:
        return SemanticChunker(
            embedder,
            similarity_threshold=similarity_threshold,
            max_chunk_words=max_chunk_words,
            min_chunk_words=min_chunk_words,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")
