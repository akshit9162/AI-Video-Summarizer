import hashlib
import json
import os

from faster_whisper import WhisperModel

CACHE_DIR = os.path.join(".cache", "speech")
TRANSCRIPT_DIR = os.path.join(CACHE_DIR, "transcripts")

_wm = None


def _ensure_dirs():
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get_whisper_model():
    global _wm
    if _wm is None:
        _wm = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _wm


def _transcribe(path):
    model = _get_whisper_model()
    segments, _ = model.transcribe(path, beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


def speech_transcript(path):
    _ensure_dirs()
    file_hash = _sha256_file(path)
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{file_hash}.json")

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            text = json.load(f)["transcript"]
    else:
        text = _transcribe(path)
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump({"transcript": text}, f)

    return text


def speech_summary(path):
    # Backward-compatible alias for existing call sites.
    return speech_transcript(path)