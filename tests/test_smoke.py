"""
Smoke tests — fast, no GPU, no network.

Markers:
  embedding  — requires sentence-transformers model download (~25 MB); skipped in CI unless
               RUN_EMBEDDING_TESTS=1 is set.
"""
import os

import pytest


# ---------------------------------------------------------------------------
# Chunker (pure Python — no heavy deps)
# ---------------------------------------------------------------------------

def test_chunker_merges_short_segments():
    from src.rag import _group_segments

    segs = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0},
    ]
    groups = _group_segments(segs)
    assert len(groups) == 1
    assert groups[0]["text"] == "hello world"
    assert groups[0]["start"] == 0.0
    assert groups[0]["end"] == 2.0


def test_chunker_splits_at_word_limit():
    from src.rag import _group_segments

    long = "word " * 150  # 150 words per segment, limit is 200
    segs = [
        {"text": long, "start": 0.0, "end": 5.0},
        {"text": long, "start": 5.0, "end": 10.0},
    ]
    groups = _group_segments(segs)
    assert len(groups) == 2
    assert groups[0]["start"] == 0.0
    assert groups[1]["start"] == 5.0


def test_chunker_empty_input():
    from src.rag import _group_segments

    assert _group_segments([]) == []


# ---------------------------------------------------------------------------
# Timestamp formatter (pure Python)
# ---------------------------------------------------------------------------

def test_fmt_seconds():
    from src.rag import _fmt

    assert _fmt(0) == "0:00"
    assert _fmt(59) == "0:59"
    assert _fmt(60) == "1:00"
    assert _fmt(65) == "1:05"
    assert _fmt(3600) == "1:00:00"
    assert _fmt(3661) == "1:01:01"


# ---------------------------------------------------------------------------
# Health endpoint (FastAPI TestClient — no heavy ML deps needed)
# ---------------------------------------------------------------------------

def test_health_endpoint():
    from dotenv import load_dotenv
    load_dotenv()
    from fastapi.testclient import TestClient
    from app.api import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_rag_query_404_for_unknown_hash():
    from dotenv import load_dotenv
    load_dotenv()
    from fastapi.testclient import TestClient
    from app.api import app

    client = TestClient(app)
    resp = client.post("/rag/query", json={"video_hash": "nonexistent", "question": "test"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# RAG index + query round-trip (requires sentence-transformers model download)
# ---------------------------------------------------------------------------

_run_embedding = os.getenv("RUN_EMBEDDING_TESTS", "0") == "1"

@pytest.mark.skipif(not _run_embedding, reason="Set RUN_EMBEDDING_TESTS=1 to run embedding tests")
def test_rag_index_and_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.rag import index_transcript, is_indexed, query_video

    segs = [
        {"text": "Deep learning is a subset of machine learning.", "start": 0.0, "end": 3.0},
        {"text": "Neural networks learn representations from raw data.", "start": 3.0, "end": 6.0},
    ]

    assert not is_indexed("testhash")
    index_transcript("testhash", segs)
    assert is_indexed("testhash")

    result = query_video("testhash", "What is deep learning?")
    assert len(result["sources"]) > 0
    assert result["sources"][0]["start"] == 0.0
