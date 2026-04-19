import pytest
from rag import RAGEngine

@pytest.fixture(scope="module")
def engine():
    return RAGEngine()

def test_init_creates_collection(engine):
    assert engine.collection is not None
    assert engine.collection.count() == 0

def test_chunk_text_basic(engine):
    text = "あ" * 1200
    chunks = engine._chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500

def test_chunk_text_short(engine):
    text = "短いテキスト"
    chunks = engine._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "短いテキスト"

def test_chunk_text_overlap(engine):
    text = "A" * 600
    chunks = engine._chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 2
    assert chunks[1][:100] == chunks[0][400:]
