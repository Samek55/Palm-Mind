import pytest

from app.services.chunking import FixedSizeChunker, SemanticChunker, split_sentences


def test_fixed_chunker_respects_size_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    chunks = FixedSizeChunker(chunk_size=10, overlap=3).split(text)

    # step is 10 - 3 = 7, so chunks start at w0, w7, w14, w21; the last one is short
    assert [len(c.split()) for c in chunks] == [10, 10, 10, 4]
    # last 3 words of a chunk start the next one
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "w24"


def test_fixed_chunker_short_text_is_single_chunk() -> None:
    assert FixedSizeChunker(chunk_size=50, overlap=10).split("just a few words") == ["just a few words"]


def test_fixed_chunker_empty_text() -> None:
    assert FixedSizeChunker().split("   ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_fixed_chunker_rejects_bad_config(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=size, overlap=overlap)


def test_split_sentences() -> None:
    text = "First sentence. Second one!  Third?\n\nNew paragraph"
    assert split_sentences(text) == ["First sentence.", "Second one!", "Third?", "New paragraph"]


def test_semantic_chunker_splits_on_topic_change(embedder) -> None:  # type: ignore[no-untyped-def]
    text = (
        "Cats are small pets. Cats are small and playful pets. "
        "Rockets fly to space quickly. Rockets fly to orbit and space."
    )
    chunks = SemanticChunker(
        embedder, similarity_threshold=0.3, max_chunk_words=100, min_chunk_words=0
    ).split(text)

    assert len(chunks) == 2
    assert "Cats" in chunks[0] and "Rockets" not in chunks[0]
    assert "Rockets" in chunks[1]


def test_semantic_chunker_respects_max_words(embedder) -> None:  # type: ignore[no-untyped-def]
    text = " ".join("Cats are small pets." for _ in range(10))
    chunks = SemanticChunker(embedder, similarity_threshold=0.0, max_chunk_words=8).split(text)

    assert all(len(c.split()) <= 8 for c in chunks)
    assert len(chunks) == 5


def test_semantic_chunker_min_words(embedder) -> None:  # type: ignore[no-untyped-def]
    text = (
        "Cats are small pets. Cats are small and playful pets. "
        "Rockets fly to space quickly. Rockets fly to orbit and space."
    )
    chunks = SemanticChunker(
        embedder, similarity_threshold=0.3, max_chunk_words=100, min_chunk_words=40
    ).split(text)

    # the topic changes, but the first chunk is too short to close yet
    assert len(chunks) == 1
