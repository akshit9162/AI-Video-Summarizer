import hashlib
import json
import os
from typing import Dict, List, Tuple

from faster_whisper import WhisperModel

CACHE_DIR = os.path.join(".cache", "speech")
TRANSCRIPT_DIR = os.path.join(CACHE_DIR, "transcripts")

_wm = None


def _ensure_dirs():
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get_whisper_model() -> WhisperModel:
    global _wm
    if _wm is None:
        _wm = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _wm


def _transcribe(path: str) -> Tuple[str, List[Dict]]:
    """Returns (full_text, [{text, start, end}])."""
    model = _get_whisper_model()
    raw_segments, _ = model.transcribe(path, beam_size=1, vad_filter=True)
    segments = []
    texts = []
    for seg in raw_segments:
        t = seg.text.strip()
        if t:
            segments.append({"text": t, "start": seg.start, "end": seg.end})
            texts.append(t)
    return " ".join(texts), segments


def file_hash(path: str) -> str:
    return _sha256_file(path)


def speech_transcript_with_meta(path: str) -> Tuple[str, str, List[Dict]]:
    """
    Returns (transcript_text, video_hash, segments).
    Segments: [{text, start, end}] — used for RAG indexing.
    Triggers RAG indexing as a side-effect (idempotent).
    """
    _ensure_dirs()
    video_hash = _sha256_file(path)
    cache_path = os.path.join(TRANSCRIPT_DIR, f"{video_hash}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data["transcript"]
        segments = data.get("segments", [])
        if not segments:
            # Old cache without segments — re-transcribe to capture timestamps
            text, segments = _transcribe(path)
            data["segments"] = segments
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
    else:
        text, segments = _transcribe(path)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"transcript": text, "segments": segments}, f)

    # Index into LlamaIndex (idempotent — skips if already done)
    if segments:
        try:
            from src.rag import index_transcript
            index_transcript(video_hash, segments)
        except Exception as exc:
            print(f"[rag] indexing skipped: {exc}")

    return text, video_hash, segments


def speech_transcript(path: str) -> str:
    """Backward-compatible: returns transcript text only."""
    text, _, _ = speech_transcript_with_meta(path)
    return text


def speech_summary(path: str) -> str:
    return speech_transcript(path)
