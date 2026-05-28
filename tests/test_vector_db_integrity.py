"""Unit tests for FAISS vector database integrity (freshness, geography, consistency)."""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import pickle
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).parent.parent

JSON_PATH     = ROOT_DIR / os.environ.get("JSON_PATH", "events_vectorized.json")
INDEX_DIR     = ROOT_DIR / os.environ.get("INDEX_DIR", "faiss_index")
PKL_PATH      = INDEX_DIR / "index.pkl"
API_REGION    = os.environ.get("API_REGION", "Île-de-France")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))
API_LANG      = os.environ.get("API_LANG", "fr")
API_LIMIT     = int(os.environ.get("API_LIMIT", 100))
API_KEYWORDS  = os.environ.get("API_KEYWORDS", "en physique")


def _now_ms() -> int:
    return int(datetime.datetime.now().timestamp() * 1000)


def _cutoff_ms() -> int:
    return int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp() * 1000)


def _load_json_chunks() -> list[dict]:
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_faiss_documents():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def _import_load_events():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    mock = MagicMock()
    with patch.dict("sys.modules", {
        "faiss": mock, "mistralai": mock, "mistralai.client": mock, "vector_store": mock,
    }):
        spec = importlib.util.spec_from_file_location("_vai_test", ROOT_DIR / "vectorize_and_index.py")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
    return getattr(module, "load_events", None)


class TestJSONSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not JSON_PATH.exists():
            raise unittest.SkipTest(f"JSON file not found: {JSON_PATH}")
        cls.chunks = _load_json_chunks()
        cls.cutoff_ms = _cutoff_ms()
        cls.now_ms = _now_ms()

    def test_non_empty(self):
        self.assertGreater(len(self.chunks), 0)

    def test_all_chunks_have_date_start(self):
        missing = [c.get("title", f"#{i}") for i, c in enumerate(self.chunks) if "date_start" not in c]
        self.assertEqual(missing, [])

    def test_date_start_is_int_or_null(self):
        bad = [c.get("title") for c in self.chunks
               if c.get("date_start") is not None and not isinstance(c["date_start"], int)]
        self.assertEqual(bad, [])

    def test_dated_events_within_one_year(self):
        stale = [c.get("title") for c in self.chunks
                 if c.get("date_start") is not None and c["date_start"] < self.cutoff_ms]
        self.assertEqual(stale, [], f"{len(stale)} events older than 365 days")

    def test_dated_events_not_far_future(self):
        far_limit = self.now_ms + int(datetime.timedelta(days=365).total_seconds() * 1000)
        far = [c.get("title") for c in self.chunks
               if c.get("date_start") is not None and c["date_start"] > far_limit]
        self.assertEqual(far, [], f"{len(far)} events with suspiciously far-future dates")

    def test_all_chunks_have_location(self):
        missing = [c.get("title", f"#{i}") for i, c in enumerate(self.chunks) if "location" not in c]
        self.assertEqual(missing, [])

    def test_location_is_non_empty_string(self):
        invalid = [c.get("title") for c in self.chunks
                   if not isinstance(c.get("location"), str) or not c["location"].strip()]
        self.assertEqual(invalid, [])

    def test_no_inconnu_location(self):
        inconnu = [c.get("title") for c in self.chunks if c.get("location") == "Inconnu"]
        self.assertEqual(inconnu, [], f"{len(inconnu)} events with placeholder location")


class TestFAISSIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PKL_PATH.exists():
            raise unittest.SkipTest(f"FAISS pickle not found: {PKL_PATH}")
        cls.docs = _load_faiss_documents()
        cls.cutoff_ms = _cutoff_ms()

    def test_non_empty(self):
        self.assertGreater(len(self.docs), 0)

    def test_dated_docs_within_one_year(self):
        stale = [doc.metadata.get("title") for doc in self.docs
                 if doc.metadata.get("date_start_ms") is not None
                 and doc.metadata["date_start_ms"] < self.cutoff_ms]
        self.assertEqual(stale, [], f"{len(stale)} documents older than 365 days")

    def test_date_strings_format(self):
        for doc in self.docs:
            date_str = doc.metadata.get("date_start", "")
            if date_str == "Date inconnue":
                continue
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except ValueError:
                self.fail(f"Unparseable date_start {date_str!r} for {doc.metadata.get('title')!r}")

    def test_date_string_and_ms_consistent(self):
        for doc in self.docs:
            ms = doc.metadata.get("date_start_ms")
            s = doc.metadata.get("date_start", "Date inconnue")
            if ms is None or s == "Date inconnue":
                continue
            expected = datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
            self.assertEqual(expected, s, f"Timestamp mismatch for {doc.metadata.get('title')!r}")

    def test_all_docs_have_location(self):
        missing = [doc.metadata.get("title", f"#{i}") for i, doc in enumerate(self.docs)
                   if "location" not in doc.metadata]
        self.assertEqual(missing, [])

    def test_location_is_non_empty_string(self):
        invalid = [doc.metadata.get("title", "<no title>") for doc in self.docs
                   if not isinstance(doc.metadata.get("location"), str)
                   or not doc.metadata["location"].strip()]
        self.assertEqual(invalid, [])

    def test_no_inconnu_location(self):
        inconnu = [doc.metadata.get("title", "<no title>") for doc in self.docs
                   if doc.metadata.get("location") == "Inconnu"]
        self.assertEqual(inconnu, [], f"{len(inconnu)} documents with placeholder location")

    def test_all_docs_have_page_content(self):
        empty = [doc.metadata.get("title", "<no title>") for doc in self.docs
                 if not getattr(doc, "page_content", "").strip()]
        self.assertEqual(empty, [])


class TestAPIFilterParameters(unittest.TestCase):
    @staticmethod
    def _build_params(region=API_REGION, keywords=API_KEYWORDS, days=365):
        today = datetime.datetime.today()
        cutoff = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        return {
            "lang": API_LANG, "limit": API_LIMIT, "offset": 0,
            "where": (
                f'location_region like "{region}" '
                f'AND keywords_fr like "{keywords}" '
                f'AND firstdate_begin >= "{cutoff}" '
                f'AND firstdate_begin <= "{today.strftime("%Y-%m-%d")}"'
            ),
        }

    def test_region_filter(self):
        self.assertIn(f'location_region like "{API_REGION}"', self._build_params()["where"])

    def test_different_regions_produce_different_clauses(self):
        p1 = self._build_params(region="Île-de-France")
        p2 = self._build_params(region="Bretagne")
        self.assertIn("Île-de-France", p1["where"])
        self.assertIn("Bretagne", p2["where"])
        self.assertNotIn("Bretagne", p1["where"])

    def test_date_bounds_present(self):
        where = self._build_params()["where"]
        self.assertIn("firstdate_begin >=", where)
        self.assertIn("firstdate_begin <=", where)

    def test_upper_bound_is_today(self):
        today_str = datetime.datetime.today().strftime("%Y-%m-%d")
        self.assertIn(f'firstdate_begin <= "{today_str}"', self._build_params()["where"])

    def test_date_window_spans_365_days(self):
        where = self._build_params()["where"]
        dates = re.findall(r'firstdate_begin [<>]= "(\d{4}-\d{2}-\d{2})"', where)
        self.assertEqual(len(dates), 2)
        d_lower = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        d_upper = datetime.datetime.strptime(dates[1], "%Y-%m-%d")
        self.assertEqual((d_upper - d_lower).days, 365)


class TestLoadEventsFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fn = _import_load_events()
        if fn is None:
            raise unittest.SkipTest("vectorize_and_index.load_events could not be imported.")
        cls.load_events = staticmethod(fn)

    def _make_chunk(self, title="Événement test", location="Paris 18ème",
                    days_ago=30, chunk_index=0, chunk_count=1):
        date_ms = None
        if days_ago is not None:
            date_ms = int((datetime.datetime.now() - datetime.timedelta(days=days_ago)).timestamp() * 1000)
        return {
            "title": title,
            "description": f"Description pour {title}.",
            "location": location,
            "date_start": date_ms,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "vector": [0.01 * i for i in range(EMBEDDING_DIM)],
        }

    def _write_fixture(self, chunks):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as f:
            json.dump(chunks, f, ensure_ascii=False)
            path = f.name
        self.addCleanup(os.unlink, path)
        return path

    def test_returns_docs_and_vectors(self):
        docs, vectors = self.load_events(self._write_fixture([self._make_chunk()]))
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(vectors), 1)

    def test_required_metadata_keys(self):
        docs, _ = self.load_events(self._write_fixture([self._make_chunk()]))
        required = {"title", "location", "date_start", "date_start_ms", "chunk_index", "chunk_count"}
        self.assertEqual(required - set(docs[0].metadata.keys()), set())

    def test_recent_event_passes_freshness(self):
        docs, _ = self.load_events(self._write_fixture([self._make_chunk(days_ago=30)]))
        cutoff = _cutoff_ms()
        for doc in docs:
            ms = doc.metadata.get("date_start_ms")
            if ms is not None:
                self.assertGreaterEqual(ms, cutoff)

    def test_stale_event_detected(self):
        docs, _ = self.load_events(self._write_fixture([self._make_chunk(days_ago=400)]))
        for doc in docs:
            ms = doc.metadata.get("date_start_ms")
            if ms is not None:
                self.assertLess(ms, _cutoff_ms())

    def test_null_date_becomes_date_inconnue(self):
        docs, _ = self.load_events(self._write_fixture([self._make_chunk(days_ago=None)]))
        self.assertEqual(docs[0].metadata["date_start"], "Date inconnue")
        self.assertIsNone(docs[0].metadata.get("date_start_ms"))

    def test_vectors_preserved(self):
        chunk = self._make_chunk()
        _, vectors = self.load_events(self._write_fixture([chunk]))
        self.assertEqual(vectors[0], chunk["vector"])

    def test_multiple_chunks_all_loaded(self):
        chunks = [self._make_chunk("Atelier", chunk_index=i, chunk_count=3) for i in range(3)]
        docs, vectors = self.load_events(self._write_fixture(chunks))
        self.assertEqual(len(docs), 3)
        self.assertEqual(len(vectors), 3)

    def test_page_content_is_description(self):
        chunk = self._make_chunk("Conférence", days_ago=10)
        docs, _ = self.load_events(self._write_fixture([chunk]))
        self.assertEqual(docs[0].page_content, chunk["description"])

    def test_location_metadata_matches_json(self):
        chunk = self._make_chunk(location="Maison de l'Emploi de Créteil")
        docs, _ = self.load_events(self._write_fixture([chunk]))
        self.assertEqual(docs[0].metadata["location"], "Maison de l'Emploi de Créteil")


class TestDataConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [str(p) for p in (JSON_PATH, PKL_PATH) if not p.exists()]
        if missing:
            raise unittest.SkipTest(f"Missing files: {', '.join(missing)}")
        cls.chunks = _load_json_chunks()
        cls.docs = _load_faiss_documents()

    def test_doc_counts_match(self):
        self.assertEqual(
            len(self.chunks), len(self.docs),
            f"JSON has {len(self.chunks)} chunks but FAISS has {len(self.docs)} documents",
        )

    def test_chunk_indices_valid(self):
        chunks = [c for c in self.chunks if c.get("chunk_index") is not None]
        if not chunks:
            self.skipTest("No chunk_index metadata — re-run vectorize_events.py")
        invalid = [
            c for c in chunks
            if not isinstance(c.get("chunk_index"), int)
            or not isinstance(c.get("chunk_count"), int)
            or c.get("chunk_count", 0) < 1
            or not (0 <= c.get("chunk_index", -1) < c.get("chunk_count", 0))
        ]
        self.assertEqual(invalid, [])

    def test_vector_dimensions(self):
        wrong = [c.get("title") for c in self.chunks if len(c.get("vector", [])) != EMBEDDING_DIM]
        self.assertEqual(wrong, [], f"{len(wrong)} chunks with wrong vector dimension (expected {EMBEDDING_DIM})")

    def test_titles_non_empty(self):
        no_title = [i for i, c in enumerate(self.chunks) if not c.get("title", "").strip()]
        self.assertEqual(no_title, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
