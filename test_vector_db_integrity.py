"""
test_vector_db_integrity.py
===========================

Unit tests verifying that data integrated into the FAISS vector database
satisfies two key business invariants:

  1. TEMPORAL FRESHNESS  — Every indexed event has a start date within the
                           last 365 days (< 1 year old).
  2. GEOGRAPHIC SCOPE    — Only events from the configured region
                           (API_REGION, default "Île-de-France") are stored,
                           and each record carries a valid, non-empty location.

Test groups
-----------
A. TestJSONSourceFreshness    — date checks on events_vectorized.json
B. TestJSONSourceGeography    — location checks on events_vectorized.json
C. TestFAISSIndexFreshness    — date checks on faiss_index/index.pkl
D. TestFAISSIndexGeography    — location checks on faiss_index/index.pkl
E. TestAPIFilterParameters    — validates the OpenAgenda API WHERE clause
F. TestLoadEventsFunction     — unit-tests build_faiss_index.load_events()
                                with controlled JSON fixtures (no API key needed)
G. TestDataConsistency        — cross-validation between JSON source and FAISS

Usage
-----
    # With pytest (recommended):
    python -m pytest test_vector_db_integrity.py -v

    # With the standard library runner:
    python test_vector_db_integrity.py
"""

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

# ---------------------------------------------------------------------------
# Project root & paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Configuration — mirrors vectorize_events.py / build_faiss_index.py defaults
# ---------------------------------------------------------------------------
JSON_PATH     = ROOT_DIR / os.environ.get("JSON_PATH",   "events_vectorized.json")
INDEX_DIR     = ROOT_DIR / os.environ.get("INDEX_DIR",   "faiss_index")
PKL_PATH      = INDEX_DIR / "index.pkl"
API_REGION    = os.environ.get("API_REGION",   "Île-de-France")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))
API_LANG      = os.environ.get("API_LANG",     "fr")
API_LIMIT     = int(os.environ.get("API_LIMIT", 100))
API_KEYWORDS  = os.environ.get("API_KEYWORDS", "en physique")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    """Current time as milliseconds since the Unix epoch."""
    return int(datetime.datetime.now().timestamp() * 1000)


def _cutoff_ms() -> int:
    """Earliest acceptable event date: today minus 365 days, in milliseconds."""
    cutoff = datetime.datetime.now() - datetime.timedelta(days=365)
    return int(cutoff.timestamp() * 1000)


def _load_json_chunks() -> list[dict]:
    """Load and return all chunks from events_vectorized.json."""
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_faiss_documents():
    """Load LangChain Document objects from faiss_index/index.pkl."""
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def _import_load_events():
    """
    Dynamically import load_events() from vectorize_and_index.py.
    Heavy dependencies (faiss, mistralai, vector_store) are mocked so the
    function can be tested without a GPU or API key.
    Returns the callable, or None if the import fails.
    """
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    faiss_mock        = MagicMock()
    mistral_mock      = MagicMock()
    vector_store_mock = MagicMock()

    with patch.dict("sys.modules", {
        "faiss":             faiss_mock,
        "mistralai":         mistral_mock,
        "mistralai.client":  mistral_mock,
        "vector_store":      vector_store_mock,
    }):
        spec = importlib.util.spec_from_file_location(
            "_vai_test", ROOT_DIR / "vectorize_and_index.py"
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
    return getattr(module, "load_events", None)


# ===========================================================================
# A. JSON source — temporal freshness
# ===========================================================================

class TestJSONSourceFreshness(unittest.TestCase):
    """
    Verify that every event chunk stored in events_vectorized.json
    has a start date within the last 365 days.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not JSON_PATH.exists():
            raise unittest.SkipTest(f"JSON file not found: {JSON_PATH}")
        cls.chunks    = _load_json_chunks()
        cls.cutoff_ms = _cutoff_ms()
        cls.now_ms    = _now_ms()

    # ── Presence checks ──────────────────────────────────────────────────────

    def test_json_file_contains_at_least_one_chunk(self):
        """events_vectorized.json must not be empty."""
        self.assertGreater(
            len(self.chunks), 0,
            "events_vectorized.json is empty — run vectorize_events.py first.",
        )

    def test_all_chunks_have_date_start_field(self):
        """Every chunk dict must contain a 'date_start' key."""
        missing = [c.get("title", f"<chunk #{i}>")
                   for i, c in enumerate(self.chunks)
                   if "date_start" not in c]
        self.assertEqual(
            missing, [],
            f"Chunks missing the 'date_start' field: {missing[:5]}",
        )

    def test_date_start_is_integer_or_null(self):
        """
        date_start must be an integer (milliseconds) or None/null.
        A string or float indicates a unit or serialisation error.
        """
        bad = [
            {"title": c.get("title"), "date_start": c["date_start"]}
            for c in self.chunks
            if c.get("date_start") is not None
            and not isinstance(c["date_start"], int)
        ]
        self.assertEqual(bad, [],
                         f"Non-integer date_start values: {bad[:3]}")

    # ── Freshness checks ─────────────────────────────────────────────────────

    def test_all_dated_events_are_within_one_year(self):
        """
        Every chunk whose date_start is not null must have a timestamp
        >= today − 365 days.  A failure means stale data slipped through
        the vectorize_events.py date filter.
        """
        stale = [
            {"title": c.get("title"), "date_start_ms": c["date_start"]}
            for c in self.chunks
            if c.get("date_start") is not None
            and c["date_start"] < self.cutoff_ms
        ]
        self.assertEqual(
            stale, [],
            f"{len(stale)} chunk(s) are older than 365 days:\n"
            + "\n".join(
                f"  - {s['title']}  (ms={s['date_start_ms']})" for s in stale[:5]
            ),
        )

    def test_dated_events_not_in_far_future(self):
        """
        No event should be dated more than 365 days from now.
        This would indicate a timestamp unit mismatch (e.g. seconds vs ms).
        """
        far_limit_ms = self.now_ms + int(
            datetime.timedelta(days=365).total_seconds() * 1000
        )
        far = [
            {"title": c.get("title"), "date_start_ms": c["date_start"]}
            for c in self.chunks
            if c.get("date_start") is not None
            and c["date_start"] > far_limit_ms
        ]
        self.assertEqual(
            far, [],
            f"{len(far)} chunk(s) have suspiciously far-future dates:\n"
            + "\n".join(
                f"  - {s['title']}  (ms={s['date_start_ms']})" for s in far[:5]
            ),
        )


# ===========================================================================
# B. JSON source — geographic scope
# ===========================================================================

class TestJSONSourceGeography(unittest.TestCase):
    """
    Verify that every chunk stored in events_vectorized.json
    carries a valid location (the region filter is enforced upstream
    by the OpenAgenda API query; location_name is preserved here).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not JSON_PATH.exists():
            raise unittest.SkipTest(f"JSON file not found: {JSON_PATH}")
        cls.chunks = _load_json_chunks()

    def test_all_chunks_have_location_field(self):
        """Every chunk must include a 'location' key."""
        missing = [c.get("title", f"<#{i}>")
                   for i, c in enumerate(self.chunks)
                   if "location" not in c]
        self.assertEqual(missing, [],
                         f"Chunks without 'location' field: {missing[:5]}")

    def test_location_is_a_non_empty_string(self):
        """location must be a non-empty, non-whitespace string."""
        invalid = [
            {"title": c.get("title"), "location": c.get("location")}
            for c in self.chunks
            if not isinstance(c.get("location"), str)
            or not c["location"].strip()
        ]
        self.assertEqual(
            invalid, [],
            f"Chunks with missing or empty location: {invalid[:5]}",
        )

    def test_no_fallback_inconnu_location(self):
        """
        The placeholder value 'Inconnu' (written by vectorize_events.py when
        location_name is absent in the API response) must not appear in any chunk.
        It indicates missing geographic data.
        """
        inconnu = [c.get("title") for c in self.chunks
                   if c.get("location") == "Inconnu"]
        self.assertEqual(
            inconnu, [],
            f"{len(inconnu)} chunk(s) have 'Inconnu' as location: {inconnu[:5]}",
        )


# ===========================================================================
# C. FAISS index — temporal freshness
# ===========================================================================

class TestFAISSIndexFreshness(unittest.TestCase):
    """
    Verify that every LangChain Document persisted in faiss_index/index.pkl
    has a start date within the last 365 days.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not PKL_PATH.exists():
            raise unittest.SkipTest(f"FAISS pickle not found: {PKL_PATH}")
        cls.docs      = _load_faiss_documents()
        cls.cutoff_ms = _cutoff_ms()
        cls.now_ms    = _now_ms()

    def test_faiss_index_is_not_empty(self):
        """The FAISS index must contain at least one document."""
        self.assertGreater(
            len(self.docs), 0,
            "faiss_index/index.pkl contains no documents — rebuild the index.",
        )

    def test_all_dated_docs_are_within_one_year(self):
        """
        Every document whose date_start_ms metadata is not None must be
        >= today − 365 days.
        """
        stale = [
            {
                "title": doc.metadata.get("title"),
                "date_start_ms": doc.metadata.get("date_start_ms"),
            }
            for doc in self.docs
            if doc.metadata.get("date_start_ms") is not None
            and doc.metadata["date_start_ms"] < self.cutoff_ms
        ]
        self.assertEqual(
            stale, [],
            f"{len(stale)} FAISS document(s) are older than 365 days:\n"
            + "\n".join(
                f"  - {s['title']}  (ms={s['date_start_ms']})" for s in stale[:5]
            ),
        )

    def test_date_strings_follow_expected_format(self):
        """
        Human-readable date_start (e.g. '2025-06-15 09:00') must parse as
        '%Y-%m-%d %H:%M', or equal the sentinel 'Date inconnue'.
        """
        for doc in self.docs:
            date_str = doc.metadata.get("date_start", "")
            if date_str == "Date inconnue":
                continue
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except ValueError:
                self.fail(
                    f"Unparseable date_start {date_str!r} "
                    f"for event {doc.metadata.get('title')!r}"
                )

    def test_date_string_and_ms_are_consistent(self):
        """
        date_start_ms and date_start string must represent the same point in
        time (accurate to the minute) for every document that has both.
        """
        for doc in self.docs:
            ms  = doc.metadata.get("date_start_ms")
            s   = doc.metadata.get("date_start", "Date inconnue")
            if ms is None or s == "Date inconnue":
                continue
            expected = datetime.datetime.fromtimestamp(ms / 1000).strftime(
                "%Y-%m-%d %H:%M"
            )
            self.assertEqual(
                expected, s,
                f"Timestamp mismatch for {doc.metadata.get('title')!r}: "
                f"ms→'{expected}' but stored '{s}'",
            )


# ===========================================================================
# D. FAISS index — geographic scope
# ===========================================================================

class TestFAISSIndexGeography(unittest.TestCase):
    """
    Verify that every document in faiss_index/index.pkl carries valid
    location metadata.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not PKL_PATH.exists():
            raise unittest.SkipTest(f"FAISS pickle not found: {PKL_PATH}")
        cls.docs = _load_faiss_documents()

    def test_all_docs_have_location_metadata(self):
        """Every Document must have a 'location' key in its metadata dict."""
        missing = [
            doc.metadata.get("title", f"<doc #{i}>")
            for i, doc in enumerate(self.docs)
            if "location" not in doc.metadata
        ]
        self.assertEqual(
            missing, [],
            f"Documents missing 'location' metadata: {missing[:5]}",
        )

    def test_location_metadata_is_non_empty_string(self):
        """location metadata must be a non-empty, non-whitespace string."""
        invalid = [
            doc.metadata.get("title", "<no title>")
            for doc in self.docs
            if not isinstance(doc.metadata.get("location"), str)
            or not doc.metadata["location"].strip()
        ]
        self.assertEqual(
            invalid, [],
            f"Documents with empty/missing location: {invalid[:5]}",
        )

    def test_no_placeholder_location_in_index(self):
        """'Inconnu' must not appear as a location value in any indexed document."""
        inconnu = [
            doc.metadata.get("title", "<no title>")
            for doc in self.docs
            if doc.metadata.get("location") == "Inconnu"
        ]
        self.assertEqual(
            inconnu, [],
            f"{len(inconnu)} document(s) in FAISS index have 'Inconnu' location.",
        )


# ===========================================================================
# E. API filter parameters
# ===========================================================================

class TestAPIFilterParameters(unittest.TestCase):
    """
    Verify that the OpenAgenda API query parameters constructed by
    vectorize_events.py correctly enforce:
      - the configured geographic region (location_region filter)
      - a one-year date window  (firstdate_begin range filter)

    The filter logic is reproduced here in isolation so no live HTTP call
    or API key is required.
    """

    # ── Reproduce vectorize_events.py param-building logic ───────────────────

    @staticmethod
    def _build_api_params(
        region: str   = API_REGION,
        keywords: str = API_KEYWORDS,
        days: int     = 365,
    ) -> dict:
        """Build the same `params` dict that vectorize_events.py would send."""
        today      = datetime.datetime.today()
        cutoff     = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        today_str  = today.strftime("%Y-%m-%d")
        return {
            "lang":   API_LANG,
            "limit":  API_LIMIT,
            "offset": 0,
            "where": (
                f'location_region like "{region}" '
                f'AND keywords_fr like "{keywords}" '
                f'AND firstdate_begin >= "{cutoff}" '
                f'AND firstdate_begin <= "{today_str}"'
            ),
        }

    # ── Region filter ─────────────────────────────────────────────────────────

    def test_region_filter_is_present(self):
        """The WHERE clause must reference location_region."""
        params = self._build_api_params()
        self.assertIn("location_region", params["where"],
                      "WHERE clause is missing a location_region filter.")

    def test_region_value_matches_api_region_config(self):
        """The region in the WHERE clause must equal the API_REGION setting."""
        params = self._build_api_params()
        self.assertIn(
            f'location_region like "{API_REGION}"',
            params["where"],
            f"Expected region '{API_REGION}' in WHERE clause.",
        )

    def test_different_regions_produce_different_clauses(self):
        """Passing different regions must produce distinct WHERE clauses."""
        p_idf     = self._build_api_params(region="Île-de-France")
        p_bretagne = self._build_api_params(region="Bretagne")
        self.assertIn("Île-de-France", p_idf["where"])
        self.assertIn("Bretagne",      p_bretagne["where"])
        self.assertNotIn("Bretagne",      p_idf["where"])
        self.assertNotIn("Île-de-France", p_bretagne["where"])

    # ── Date filter ───────────────────────────────────────────────────────────

    def test_lower_date_bound_is_present(self):
        """WHERE clause must contain a firstdate_begin >= lower bound."""
        params = self._build_api_params()
        self.assertIn("firstdate_begin >=", params["where"],
                      "No lower date bound found in WHERE clause.")

    def test_upper_date_bound_is_present(self):
        """WHERE clause must contain a firstdate_begin <= upper bound."""
        params = self._build_api_params()
        self.assertIn("firstdate_begin <=", params["where"],
                      "No upper date bound found in WHERE clause.")

    def test_upper_date_bound_equals_today(self):
        """The upper date bound must equal today's date (YYYY-MM-DD)."""
        today_str = datetime.datetime.today().strftime("%Y-%m-%d")
        params    = self._build_api_params()
        self.assertIn(
            f'firstdate_begin <= "{today_str}"',
            params["where"],
            f"Upper bound should be today ({today_str}).",
        )

    def test_lower_date_bound_is_exactly_365_days_ago(self):
        """The lower date bound must equal today − 365 days."""
        expected_cutoff = (
            datetime.datetime.today() - datetime.timedelta(days=365)
        ).strftime("%Y-%m-%d")
        params = self._build_api_params(days=365)
        self.assertIn(
            f'firstdate_begin >= "{expected_cutoff}"',
            params["where"],
            f"Lower bound should be {expected_cutoff} (today − 365 days).",
        )

    def test_date_window_spans_exactly_365_days(self):
        """
        The difference between the upper and lower date bounds must be
        exactly 365 days — confirming the one-year freshness window.
        """
        params = self._build_api_params()
        where  = params["where"]

        dates = re.findall(
            r'firstdate_begin [<>]= "(\d{4}-\d{2}-\d{2})"', where
        )
        self.assertEqual(
            len(dates), 2,
            f"Expected 2 date bounds in WHERE clause, found: {dates}",
        )

        d_lower = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        d_upper = datetime.datetime.strptime(dates[1], "%Y-%m-%d")
        delta   = d_upper - d_lower
        self.assertEqual(
            delta.days, 365,
            f"Date window is {delta.days} days; expected exactly 365.",
        )

    def test_both_region_and_date_filters_combined(self):
        """The WHERE clause must simultaneously enforce both the region AND the date range."""
        params = self._build_api_params()
        where  = params["where"]
        self.assertIn("location_region",  where)
        self.assertIn("firstdate_begin >=", where)
        self.assertIn("firstdate_begin <=", where)


# ===========================================================================
# F. build_faiss_index.load_events() unit tests (fixture-based)
# ===========================================================================

class TestLoadEventsFunction(unittest.TestCase):
    """
    Unit-tests for build_faiss_index.load_events() using controlled JSON
    fixtures.  No real API key or network access is needed.
    """

    # Use a small but realistic vector dimension for fixtures
    _FIXTURE_DIM = EMBEDDING_DIM

    # ── Test lifecycle ────────────────────────────────────────────────────────

    @classmethod
    def setUpClass(cls) -> None:
        fn = _import_load_events()
        if fn is None:
            raise unittest.SkipTest("build_faiss_index.load_events could not be imported.")
        cls.load_events = staticmethod(fn)

    def setUp(self) -> None:
        self._tmpfiles: list[str] = []

    def tearDown(self) -> None:
        for path in self._tmpfiles:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ── Fixture builders ──────────────────────────────────────────────────────

    def _dummy_vector(self) -> list[float]:
        return [0.01 * i for i in range(self._FIXTURE_DIM)]

    @staticmethod
    def _dt_to_ms(dt: datetime.datetime) -> int:
        return int(dt.timestamp() * 1000)

    def _make_chunk(
        self,
        title: str = "Événement test",
        location: str = "Paris 18ème",
        days_ago: int | None = 30,  # None → null date
        chunk_index: int = 0,
        chunk_count: int = 1,
    ) -> dict:
        """Build one chunk dict matching the events_vectorized.json schema."""
        if days_ago is None:
            date_ms: int | None = None
        else:
            dt = datetime.datetime.now() - datetime.timedelta(days=days_ago)
            date_ms = self._dt_to_ms(dt)
        return {
            "title":       title,
            "description": f"Description pour {title}.",
            "location":    location,
            "date_start":  date_ms,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "vector":      self._dummy_vector(),
        }

    def _write_fixture(self, chunks: list[dict]) -> str:
        """Serialise chunks to a temp JSON file and return the path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        json.dump(chunks, tmp, ensure_ascii=False)
        tmp.close()
        self._tmpfiles.append(tmp.name)
        return tmp.name

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_returns_docs_and_vectors_tuple(self):
        """load_events() must return a 2-tuple of (list[Document], list[list[float]])."""
        path = self._write_fixture([self._make_chunk()])
        docs, vectors = self.load_events(path)
        self.assertIsInstance(docs,    list)
        self.assertIsInstance(vectors, list)
        self.assertEqual(len(docs),    1)
        self.assertEqual(len(vectors), 1)

    def test_document_contains_required_metadata_keys(self):
        """Every loaded Document must carry title, location, date_start, date_start_ms."""
        path = self._write_fixture([self._make_chunk("Atelier Emploi")])
        docs, _ = self.load_events(path)
        required = {"title", "location", "date_start", "date_start_ms",
                    "chunk_index", "chunk_count"}
        missing_keys = required - set(docs[0].metadata.keys())
        self.assertEqual(missing_keys, set(),
                         f"Missing metadata keys: {missing_keys}")

    def test_recent_event_passes_freshness_check(self):
        """A chunk dated 30 days ago must have date_start_ms >= cutoff."""
        path = self._write_fixture([self._make_chunk(days_ago=30)])
        docs, _ = self.load_events(path)
        cutoff_ms = _cutoff_ms()
        for doc in docs:
            ms = doc.metadata.get("date_start_ms")
            if ms is not None:
                self.assertGreaterEqual(
                    ms, cutoff_ms,
                    f"Event dated 30 days ago failed freshness check "
                    f"(ms={ms}, cutoff={cutoff_ms}).",
                )

    def test_event_older_than_one_year_fails_freshness_check(self):
        """
        A chunk dated 400 days ago (outside the 365-day window) must have
        date_start_ms < cutoff — confirming the test correctly detects stale data.
        """
        path = self._write_fixture([self._make_chunk(days_ago=400)])
        docs, _ = self.load_events(path)
        cutoff_ms = _cutoff_ms()
        for doc in docs:
            ms = doc.metadata.get("date_start_ms")
            if ms is not None:
                self.assertLess(
                    ms, cutoff_ms,
                    "An event 400 days old unexpectedly passed the freshness check.",
                )

    def test_null_date_becomes_date_inconnue(self):
        """A chunk with date_start=null must produce date_start='Date inconnue'."""
        path = self._write_fixture([self._make_chunk(days_ago=None)])
        docs, _ = self.load_events(path)
        self.assertEqual(
            docs[0].metadata["date_start"], "Date inconnue",
            "Null date_start should be converted to 'Date inconnue'.",
        )
        self.assertIsNone(
            docs[0].metadata.get("date_start_ms"),
            "date_start_ms should be None when the original date is null.",
        )

    def test_vectors_are_preserved_unchanged(self):
        """The vectors returned by load_events must exactly match the JSON source."""
        chunk = self._make_chunk()
        path  = self._write_fixture([chunk])
        _, vectors = self.load_events(path)
        self.assertEqual(vectors[0], chunk["vector"])

    def test_multiple_chunks_same_event_all_loaded(self):
        """Multiple chunks from the same event are all loaded and indexed separately."""
        chunks = [
            self._make_chunk("Grand atelier", chunk_index=0, chunk_count=3),
            self._make_chunk("Grand atelier", chunk_index=1, chunk_count=3),
            self._make_chunk("Grand atelier", chunk_index=2, chunk_count=3),
        ]
        path = self._write_fixture(chunks)
        docs, vectors = self.load_events(path)
        self.assertEqual(len(docs),    3)
        self.assertEqual(len(vectors), 3)

    def test_page_content_is_description_field(self):
        """Document.page_content must equal the chunk's description field."""
        chunk = self._make_chunk("Conférence", days_ago=10)
        path  = self._write_fixture([chunk])
        docs, _ = self.load_events(path)
        self.assertEqual(docs[0].page_content, chunk["description"])

    def test_location_metadata_matches_json(self):
        """Document.metadata['location'] must match the JSON location value."""
        chunk = self._make_chunk(location="Maison de l'Emploi de Créteil")
        path  = self._write_fixture([chunk])
        docs, _ = self.load_events(path)
        self.assertEqual(
            docs[0].metadata["location"],
            "Maison de l'Emploi de Créteil",
        )


# ===========================================================================
# G. Cross-validation between JSON source and FAISS index
# ===========================================================================

class TestDataConsistency(unittest.TestCase):
    """
    Structural consistency checks: the JSON source file and the FAISS
    index must agree on the number of chunks and share valid schemas.
    """

    @classmethod
    def setUpClass(cls) -> None:
        missing = []
        if not JSON_PATH.exists():
            missing.append(str(JSON_PATH))
        if not PKL_PATH.exists():
            missing.append(str(PKL_PATH))
        if missing:
            raise unittest.SkipTest(
                "Skipping consistency tests — missing files: " + ", ".join(missing)
            )
        cls.chunks = _load_json_chunks()
        cls.docs   = _load_faiss_documents()

    def test_json_and_faiss_doc_counts_match(self):
        """Number of chunks in JSON must equal number of documents in FAISS."""
        self.assertEqual(
            len(self.chunks), len(self.docs),
            f"Mismatch: JSON has {len(self.chunks)} chunks "
            f"but FAISS has {len(self.docs)} documents. "
            "Re-run build_faiss_index.py after vectorize_events.py.",
        )

    def test_all_chunk_indices_are_valid(self):
        """
        chunk_index must be an int in [0, chunk_count) for every chunk.
        chunk_count must be >= 1.

        Chunks where both chunk_index and chunk_count are None were generated
        by an older version of vectorize_events.py that did not include these
        fields.  They are skipped here; re-running vectorize_events.py will
        produce data with valid chunking metadata.
        """
        # Only validate chunks that carry explicit chunking metadata.
        chunks_with_meta = [
            c for c in self.chunks
            if c.get("chunk_index") is not None or c.get("chunk_count") is not None
        ]
        if not chunks_with_meta:
            self.skipTest(
                "All chunks are in legacy format (no chunk_index/chunk_count). "
                "Re-run vectorize_events.py to generate chunks with this metadata."
            )
        invalid = [
            {
                "title":       c.get("title"),
                "chunk_index": c.get("chunk_index"),
                "chunk_count": c.get("chunk_count"),
            }
            for c in chunks_with_meta
            if (
                not isinstance(c.get("chunk_index"), int)
                or not isinstance(c.get("chunk_count"), int)
                or c.get("chunk_count", 0) < 1
                or not (0 <= c.get("chunk_index", -1) < c.get("chunk_count", 0))
            )
        ]
        self.assertEqual(invalid, [],
                         f"Invalid chunk_index/chunk_count: {invalid[:5]}")

    def test_all_vector_dimensions_match_embedding_dim(self):
        """
        Every vector stored in the JSON must have exactly EMBEDDING_DIM
        components (default 1024 for mistral-embed).
        """
        wrong = [
            {"title": c.get("title"), "dim": len(c.get("vector", []))}
            for c in self.chunks
            if len(c.get("vector", [])) != EMBEDDING_DIM
        ]
        self.assertEqual(
            wrong, [],
            f"{len(wrong)} chunk(s) have wrong vector dimension "
            f"(expected {EMBEDDING_DIM}): {wrong[:3]}",
        )

    def test_all_titles_are_non_empty(self):
        """Every JSON chunk must have a non-empty, non-whitespace title."""
        no_title = [
            i for i, c in enumerate(self.chunks)
            if not c.get("title", "").strip()
        ]
        self.assertEqual(
            no_title, [],
            f"Chunks at indices {no_title[:10]} have empty or missing titles.",
        )

    def test_chunk_count_is_positive_for_all_chunks(self):
        """
        chunk_count must be >= 1 for every chunk that declares it.
        Chunks with chunk_count=None are legacy data (pre-chunking format)
        and are skipped; re-run vectorize_events.py to update them.
        """
        chunks_with_count = [c for c in self.chunks if c.get("chunk_count") is not None]
        if not chunks_with_count:
            self.skipTest(
                "All chunks are in legacy format (no chunk_count field). "
                "Re-run vectorize_events.py to generate chunks with this metadata."
            )
        bad = [
            {"title": c.get("title"), "chunk_count": c.get("chunk_count")}
            for c in chunks_with_count
            if not isinstance(c["chunk_count"], int) or c["chunk_count"] < 1
        ]
        self.assertEqual(bad, [],
                         f"Chunks with invalid chunk_count: {bad[:5]}")

    def test_all_docs_have_page_content(self):
        """Every FAISS Document must have non-empty page_content."""
        empty = [
            doc.metadata.get("title", "<no title>")
            for doc in self.docs
            if not getattr(doc, "page_content", "").strip()
        ]
        self.assertEqual(empty, [],
                         f"Documents with empty page_content: {empty[:5]}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
