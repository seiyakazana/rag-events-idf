"""
vectorize_and_index.py
======================
Step 2 of the RAG pipeline.

Responsibilities
----------------
1. Load the pre-processed text chunks produced by data_preprocessing.py.
2. Embed each chunk with the Mistral embedding API.
3. Persist the enriched chunks (text + vectors) to JSON_PATH.
4. Build a FAISS vector index and save it to INDEX_DIR.
5. Run a small set of test queries to validate the index.

Input
-----
    events_chunks.json   — output of data_preprocessing.py

Output
------
    events_vectorized.json   — chunks with embedding vectors attached
    faiss_index/
        index.faiss          — FAISS binary index
        index.pkl            — LangChain Document list (metadata + page_content)

Previous step
-------------
    python data_preprocessing.py
"""

import datetime
import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from mistralai.client import Mistral

from vector_store import FAISSVectorStore

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mistral-embed")
CHUNKS_PATH     = os.environ.get("CHUNKS_PATH",     "events_chunks.json")
JSON_PATH       = os.environ.get("JSON_PATH",        "events_vectorized.json")
INDEX_DIR       = os.environ.get("INDEX_DIR",        "faiss_index")

TEST_QUERIES = [
    "atelier pour créer une entreprise",
    "aide au handicap en entretien d'embauche",
    "recherche d'emploi pour les jeunes",
    "événement de recrutement en Île-de-France",
    "formation professionnelle",
]


# ── Pipeline functions ────────────────────────────────────────────────────────

def load_chunks(chunks_path: str = CHUNKS_PATH) -> list[dict]:
    """Load the pre-processed chunk list from *chunks_path*."""
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], client: Mistral) -> list[dict]:
    """
    Call the Mistral embedding API for every chunk and attach the resulting
    vector under the 'vector' key.  Returns the same list, mutated in-place.
    """
    texts = [c["description"] for c in chunks]
    resp  = client.embeddings.create(model=EMBEDDING_MODEL, inputs=texts)
    for i, item in enumerate(resp.data):
        chunks[i]["vector"] = item.embedding
    return chunks


def save_vectorized(chunks: list[dict], json_path: str = JSON_PATH) -> None:
    """Persist the embedded chunks (text + vector) to *json_path*."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_events(
    json_path: str = JSON_PATH,
) -> tuple[list[Document], list[list[float]]]:
    """
    Read the vectorised JSON and return a pair:
        (list[LangChain Document], list[embedding vectors])
    One Document per chunk; the vector is separated out for FAISS ingestion.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document]       = []
    vectors: list[list[float]] = []

    for chunk in data:
        date_ms  = chunk.get("date_start")
        date_str = (
            datetime.datetime.fromtimestamp(date_ms / 1000).strftime("%Y-%m-%d %H:%M")
            if date_ms
            else "Date inconnue"
        )
        docs.append(Document(
            page_content=chunk["description"],
            metadata={
                "title":         chunk["title"],
                "location":      chunk["location"],
                "date_start":    date_str,
                "date_start_ms": date_ms,
                "chunk_index":   chunk.get("chunk_index", 0),
                "chunk_count":   chunk.get("chunk_count", 1),
            },
        ))
        vectors.append(chunk["vector"])

    return docs, vectors


def build_index(
    docs: list[Document], vectors: list[list[float]], client: Mistral
) -> FAISSVectorStore:
    """Build a FAISS vector store from pre-computed vectors and save it locally."""
    store = FAISSVectorStore(client=client)
    store.add_documents_with_vectors(docs, vectors)
    store.save_local(INDEX_DIR)
    print(f"  Index saved to '{INDEX_DIR}/' (index.faiss + index.pkl)")
    return store


def run_test_queries(store: FAISSVectorStore, k: int = 3) -> None:
    """Run the predefined test queries and print the top-k results."""
    for query in TEST_QUERIES:
        print(f"\n  Query: {query!r}")
        results = store.similarity_search_with_score(query, k=k)
        for rank, (doc, score) in enumerate(results, 1):
            m           = doc.metadata
            snippet     = doc.page_content[:100].replace("\n", " ")
            chunk_label = f"chunk {m['chunk_index'] + 1}/{m['chunk_count']}"
            print(f"    {rank}. [{score:.4f}] {m['title']!r} ({chunk_label})")
            print(f"         {m['location']} | {m['date_start']}")
            print(f"         {snippet}...")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Step 1 — load pre-processed chunks
    print("=== Step 1: Load pre-processed chunks ===")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  {len(chunks)} chunk(s) loaded from '{CHUNKS_PATH}'.")

    # Step 2 — embed
    print("\n=== Step 2: Embed chunks with Mistral ===")
    chunks = embed_chunks(chunks, client)
    print("  Embedding complete:")
    for c in chunks:
        print(f"    [{c['chunk_index'] + 1}/{c['chunk_count']}] {c['title']}")
        print(f"      dim={len(c['vector'])}, first values={c['vector'][:4]}")

    # Step 3 — save vectorised JSON
    print(f"\n=== Step 3: Save vectorised chunks to '{JSON_PATH}' ===")
    save_vectorized(chunks, JSON_PATH)
    print(f"  Saved {len(chunks)} chunk(s).")

    # Step 4 — build FAISS index
    print("\n=== Step 4: Build FAISS index ===")
    docs, vectors = load_events(JSON_PATH)
    store = build_index(docs, vectors, client)
    print(f"  Vectors in index: {store._index.ntotal}")

    # Step 5 — test queries
    print("\n=== Step 5: Test queries ===")
    run_test_queries(store)

    # Step 6 — round-trip verification
    print("\n=== Step 6: Round-trip verification ===")
    store2  = FAISSVectorStore.load_local(INDEX_DIR, client=client)
    print(f"  Reloaded index: {store2._index.ntotal} vectors")
    results = store2.similarity_search("creation d'entreprise", k=2)
    print(f"  Round-trip query returned {len(results)} result(s):")
    for doc in results:
        print(f"    - {doc.metadata['title']!r}")

    print("\nDone. Run 'streamlit run chatbot.py' to launch the assistant.")


if __name__ == "__main__":
    main()
