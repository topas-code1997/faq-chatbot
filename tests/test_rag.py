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

def test_add_faq_stores_documents(engine):
    faqs = [
        {"q": "有給休暇の申請方法は？", "a": "マイページから申請してください。"},
        {"q": "経費精算の締め日は？", "a": "毎月末日です。"},
    ]
    engine.add_faq(faqs)
    assert engine.collection.count() == 2

def test_add_faq_replaces_existing(engine):
    faqs = [{"q": "新しい質問", "a": "新しい回答"}]
    engine.add_faq(faqs)
    assert engine.collection.count() == 1

def test_add_pdf_chunks_and_stores(engine):
    text = "社内規則の内容です。" * 100
    engine.add_pdf(text, "rules.pdf")
    assert "rules.pdf" in engine.uploaded_pdfs

def test_add_pdf_replaces_same_file(engine):
    count_before = engine.collection.count()
    text = "更新されたルールです。" * 10
    engine.add_pdf(text, "rules.pdf")
    count_after = engine.collection.count()
    assert count_after <= count_before

def test_remove_source(engine):
    engine.add_pdf("テスト文書。" * 20, "temp.pdf")
    engine.remove_source("temp.pdf")
    assert "temp.pdf" not in engine.uploaded_pdfs
    result = engine.collection.get(where={"source": "temp.pdf"})
    assert result["ids"] == []

def test_search_returns_relevant_chunks(engine):
    faqs = [{"q": "有給休暇の申請方法は？", "a": "マイページから申請してください。"}]
    engine.add_faq(faqs)
    results = engine.search("有給休暇を取りたい", top_k=1)
    assert len(results) == 1
    assert "有給" in results[0]

def test_search_empty_collection_returns_empty():
    fresh = RAGEngine()
    results = fresh.search("何か質問")
    assert results == []

def test_search_top_k_limit(engine):
    results = engine.search("申請", top_k=2)
    assert len(results) <= 2
