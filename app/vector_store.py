"""
vector_store.py
===============
Shared module — contains the FAISSVectorStore class used by both
vectorize_and_index.py (index construction) and chatbot.py (similarity search).
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from mistralai.client import Mistral

EMBEDDING_DIM        = int(os.environ.get("EMBEDDING_DIM", 1024))
EMBEDDING_MODEL      = os.environ.get("EMBEDDING_MODEL", "mistral-embed")
RELEVANCE_THRESHOLD  = float(os.environ.get("RELEVANCE_THRESHOLD", 0.82))


class FAISSVectorStore:
    def __init__(self, client: Mistral, dim: int = EMBEDDING_DIM):
        self._client = client
        self._index  = faiss.IndexFlatIP(dim)
        self._documents: list[Document] = []

    def add_documents_with_vectors(
        self, documents: list[Document], vectors: list[list[float]]
    ) -> None:
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._documents.extend(documents)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        resp = self._client.embeddings.create(model=EMBEDDING_MODEL, inputs=[query])
        vec  = np.array([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vec)
        k    = min(k, self._index.ntotal)
        D, I = self._index.search(vec, k)
        return [
            (self._documents[idx], float(score))
            for idx, score in zip(I[0], D[0])
            if idx != -1 and float(score) >= RELEVANCE_THRESHOLD
        ]

    def __len__(self) -> int:
        return self._index.ntotal

    def save_local(self, folder: str) -> None:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "index.pkl", "wb") as f:
            pickle.dump(self._documents, f)

    @classmethod
    def load_local(cls, folder: str, client: Mistral) -> "FAISSVectorStore":
        path  = Path(folder)
        store = cls(client=client)
        store._index     = faiss.read_index(str(path / "index.faiss"))
        with open(path / "index.pkl", "rb") as f:
            store._documents = pickle.load(f)
        return store
