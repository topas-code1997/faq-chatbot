import os
import re
import chromadb
from openai import OpenAI


class _OpenAIEmbeddingFunction:
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _embed(self, input):
        response = self._client.embeddings.create(
            input=input,
            model="text-embedding-3-small",
        )
        return [item.embedding for item in response.data]

    def __call__(self, input):
        return self._embed(input)

    def embed_documents(self, input):
        return self._embed(input)

    def embed_query(self, input):
        return self._embed(input)


class RAGEngine:
    def __init__(self):
        self._ef = _OpenAIEmbeddingFunction()
        self._client = chromadb.EphemeralClient()
        self.collection = self._client.create_collection(
            name="faq_rag",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
            get_or_create=True,
        )
        existing = self.collection.get()
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        self.uploaded_pdfs = []

    def _chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _safe_id(s):
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s)

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

    def add_pdf(self, text, filename):
        self.remove_source(filename)
        chunks = self._chunk_text(text)
        if not chunks:
            return
        safe = self._safe_id(filename)
        ids = [f"{safe}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "type": "pdf"} for _ in chunks]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        if filename not in self.uploaded_pdfs:
            self.uploaded_pdfs.append(filename)

    def remove_source(self, source):
        existing = self.collection.get(where={"source": source})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        if source in self.uploaded_pdfs:
            self.uploaded_pdfs.remove(source)

    def search(self, query, top_k=5):
        count = self.collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self.collection.query(query_texts=[query], n_results=n)
        return results["documents"][0]
