import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class RAGEngine:
    def __init__(self):
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._client = chromadb.EphemeralClient()
        self.collection = self._client.create_collection(
            name="faq_rag",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.uploaded_pdfs = []

    def _chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    def add_faq(self, faqs):
        existing = self.collection.get(where={"type": "faq"})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        if not faqs:
            return
        documents = [f"Q: {f['q']}\nA: {f['a']}" for f in faqs]
        ids = [f"faq_{i}" for i in range(len(faqs))]
        metadatas = [{"source": "faq", "type": "faq"} for _ in faqs]
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

    @staticmethod
    def _safe_id(s):
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s)
