import os
import re
from typing import Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def _extractive_summary(text: str, sentence_limit: int = 3) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(sentences[: max(1, sentence_limit)])


def _build_llm_chain():
    if ChatAnthropic is None or not os.getenv("ANTHROPIC_API_KEY"):
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise assistant that summarizes speech transcripts in plain language.",
            ),
            ("human", "Transcript:\n{transcript}\n\nReturn a 3-5 bullet summary."),
        ]
    )
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.2, max_tokens=512)
    return prompt | llm | RunnableLambda(lambda msg: msg.content.strip())


def build_transcript_agent_pipeline() -> RunnableLambda:
    llm_chain = _build_llm_chain()

    def _run(payload: Dict[str, str]) -> Dict[str, str]:
        transcript = _normalize_text(payload.get("transcript", ""))
        if not transcript:
            return {"summary": "", "mode": "empty"}

        if llm_chain is None:
            return {"summary": _extractive_summary(transcript), "mode": "extractive"}

        summary = llm_chain.invoke({"transcript": transcript})
        return {"summary": summary, "mode": "llm"}

    return RunnableLambda(_run)
