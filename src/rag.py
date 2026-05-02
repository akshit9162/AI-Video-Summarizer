import os
from typing import Dict, List, Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

STORAGE_BASE = os.path.join(".cache", "llamaindex")
_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_MAX_WORDS_PER_DOC = 200
_configured = False


def _configure():
    global _configured
    if _configured:
        return
    Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL)
    Settings.llm = _get_llm()
    _configured = True


def _get_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from llama_index.llms.anthropic import Anthropic
            return Anthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
        except ImportError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from llama_index.llms.openai import OpenAI
            return OpenAI(model="gpt-4o-mini", max_tokens=512)
        except ImportError:
            pass
    return None


def _persist_dir(video_hash: str) -> str:
    return os.path.join(STORAGE_BASE, video_hash)


def _group_segments(segments: List[Dict], max_words: int = _MAX_WORDS_PER_DOC) -> List[Dict]:
    """Merge Whisper segments into time-windowed chunks to preserve timestamp metadata."""
    groups, buf, buf_words = [], [], 0
    for seg in segments:
        words = len(seg["text"].split())
        if buf_words + words > max_words and buf:
            groups.append({
                "text": " ".join(s["text"] for s in buf),
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
            })
            buf, buf_words = [], 0
        buf.append(seg)
        buf_words += words
    if buf:
        groups.append({
            "text": " ".join(s["text"] for s in buf),
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
        })
    return groups


def index_transcript(video_hash: str, segments: List[Dict]) -> None:
    """Chunk, embed, and persist a video transcript. Idempotent."""
    persist_dir = _persist_dir(video_hash)
    if os.path.exists(os.path.join(persist_dir, "docstore.json")):
        return

    _configure()
    groups = _group_segments(segments)
    if not groups:
        return

    docs = [
        Document(
            text=g["text"],
            metadata={"start": round(g["start"], 1), "end": round(g["end"], 1)},
        )
        for g in groups
        if g["text"].strip()
    ]

    os.makedirs(persist_dir, exist_ok=True)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=persist_dir)


def query_video(video_hash: str, question: str, n_results: int = 4) -> Dict:
    """
    Retrieve top-n chunks relevant to the question.
    Returns:
        {
            "answer": str | None,   # synthesized answer if LLM is available
            "sources": [{text, start, end, timestamp_str, score}]
        }
    """
    persist_dir = _persist_dir(video_hash)
    if not os.path.exists(os.path.join(persist_dir, "docstore.json")):
        return {"answer": None, "sources": []}

    _configure()
    storage = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage)

    # Always retrieve source nodes
    retriever = index.as_retriever(similarity_top_k=n_results)
    nodes = retriever.retrieve(question)

    sources = []
    for node in nodes:
        meta = node.metadata
        start = meta.get("start", 0.0)
        end = meta.get("end", 0.0)
        sources.append({
            "text": node.text,
            "start": start,
            "end": end,
            "timestamp_str": f"{_fmt(start)} – {_fmt(end)}",
            "score": round(node.score, 3) if node.score else None,
        })

    # Synthesize answer if an LLM is configured
    answer: Optional[str] = None
    if Settings.llm is not None:
        engine = index.as_query_engine(similarity_top_k=n_results)
        response = engine.query(question)
        answer = str(response).strip()

    return {"answer": answer, "sources": sources}


def is_indexed(video_hash: str) -> bool:
    return os.path.exists(os.path.join(_persist_dir(video_hash), "docstore.json"))


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
