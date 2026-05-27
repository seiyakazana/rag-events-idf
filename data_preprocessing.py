"""
data_preprocessing.py
=====================
Step 1 of the RAG pipeline.

Responsibilities
----------------
1. Fetch public events from the OpenAgenda / OpenDataSoft API.
2. Print a preview of the results (replaces the old fetch_events.py utility).
3. Clean the raw records into a normalised DataFrame.
4. Split each description into overlapping text chunks.
5. Persist the chunks (without vectors) to CHUNKS_PATH (events_chunks.json).

Output
------
    events_chunks.json   — list of chunk dicts, schema:
        {
            "title":       str,
            "description": str,   # one chunk of the full description
            "location":    str,
            "date_start":  int | null,   # Unix ms timestamp
            "chunk_index": int,
            "chunk_count": int,
        }

Next step
---------
    python vectorize_and_index.py
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL       = os.environ.get("API_URL",        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/")
API_LANG      = os.environ.get("API_LANG",       "fr")
API_LIMIT     = int(os.environ.get("API_LIMIT",  100))
API_REGION    = os.environ.get("API_REGION",     "Île-de-France")
API_KEYWORDS  = os.environ.get("API_KEYWORDS",   "en physique")
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE",   500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
CHUNKS_PATH   = os.environ.get("CHUNKS_PATH",    "events_chunks.json")


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks, breaking on sentence or word boundaries."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in (". ", " "):
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks


# ── 1. Fetch events from API ──────────────────────────────────────────────────

def fetch_events() -> list[dict]:
    """
    Query the OpenAgenda API for events in the configured region
    that started within the last 365 days.
    Returns the raw list of record dicts.
    """
    today        = datetime.today()
    one_year_ago = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    today_str    = today.strftime("%Y-%m-%d")

    params = {
        "lang":   API_LANG,
        "limit":  API_LIMIT,
        "offset": 0,
        "where": (
            f'location_region like "{API_REGION}" '
            f'AND keywords_fr like "{API_KEYWORDS}" '
            f'AND firstdate_begin >= "{one_year_ago}" '
            f'AND firstdate_begin <= "{today_str}"'
        ),
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json().get("results", [])


# ── 2. Clean records into a DataFrame ────────────────────────────────────────

def clean_events(records: list[dict]) -> pd.DataFrame:
    """
    Normalise the raw API records into a tidy DataFrame with columns:
        title, description, location, date_start (datetime).
    """
    df = pd.DataFrame(records)
    df["title"]       = df["title_fr"].fillna("Sans titre")
    df["description"] = df["description_fr"].fillna(df["title"])
    df["location"]    = df.get("location_name", pd.Series()).fillna("Inconnu")
    df["date_start"]  = pd.to_datetime(
        df.get("firstdate_begin", pd.Series()), errors="coerce"
    )
    df = df[["title", "description", "location", "date_start"]].copy()
    return df.sort_values("date_start", ascending=False).reset_index(drop=True)


# ── 3. Build text chunks (no vectors) ────────────────────────────────────────

def build_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Split each event description into overlapping text chunks.
    Returns a list of chunk dicts ready to be embedded.
    No vectors are attached at this stage.
    """
    chunks: list[dict] = []
    for _, row in df.iterrows():
        text_chunks = chunk_text(row["description"])
        date_ms = (
            int(row["date_start"].timestamp() * 1000)
            if pd.notna(row["date_start"])
            else None
        )
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "title":       row["title"],
                "description": chunk,
                "location":    row["location"],
                "date_start":  date_ms,
                "chunk_index": i,
                "chunk_count": len(text_chunks),
            })
    return chunks


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    # Step 1 — fetch
    print("=== Step 1: Fetch events from API ===")
    records = fetch_events()
    if not records:
        print("No events found with these filters. Exiting.")
        return

    # Preview (replaces the old fetch_events.py output)
    print(f"  {len(records)} event(s) found in the last 12 months.\n")
    for i, rec in enumerate(records, 1):
        title     = rec.get("title_fr") or rec.get("title_en") or "No title"
        location  = rec.get("location_name") or "Unknown location"
        date_start = rec.get("firstdate_begin") or "N/A"
        print(f"  {i:>3}. {title}")
        print(f"       Location : {location}")
        print(f"       Starts   : {date_start}")
    print()

    # Step 2 — clean
    print("=== Step 2: Clean and normalise records ===")
    df = clean_events(records)
    print(f"  {len(df)} event(s) after cleaning.\n")

    # Step 3 — chunk
    print("=== Step 3: Split descriptions into chunks ===")
    chunks = build_chunks(df)
    print(f"  {len(df)} event(s) → {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")

    # Step 4 — save
    print(f"=== Step 4: Save chunks to '{CHUNKS_PATH}' ===")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(chunks)} chunk(s) from {len(df)} event(s).")
    print("\nDone. Run 'python vectorize_and_index.py' next.")


if __name__ == "__main__":
    main()
