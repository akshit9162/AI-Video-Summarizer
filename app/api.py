import asyncio
import json
import os
import tempfile
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from celery_app import celery_app
from src.agent_pipeline import build_transcript_agent_pipeline
from src.rag import is_indexed, query_video
from tasks import summarize_video_task

app = FastAPI(title="Video Summarization API", version="0.1.0")
transcript_pipeline = build_transcript_agent_pipeline()


class SummarizeYoutubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    ratio: float = Field(0.2, ge=0.05, le=0.5, description="Summary ratio")


class TranscriptInsightRequest(BaseModel):
    transcript: str = Field(..., min_length=1)


class RagQueryRequest(BaseModel):
    video_hash: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    n_results: int = Field(4, ge=1, le=10)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize/upload")
async def summarize_upload(
    file: UploadFile = File(...),
    ratio: float = 0.2,
) -> Dict[str, str]:
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 uploads are supported.")
    if ratio < 0.05 or ratio > 0.5:
        raise HTTPException(status_code=400, detail="ratio must be between 0.05 and 0.5.")

    os.makedirs("uploads", exist_ok=True)
    upload_path = os.path.join("uploads", f"{next(tempfile._get_candidate_names())}.mp4")
    content = await file.read()
    with open(upload_path, "wb") as fp:
        fp.write(content)

    task = summarize_video_task.delay("upload", upload_path, ratio)
    return {"task_id": task.id}


@app.post("/summarize/youtube")
def summarize_youtube(req: SummarizeYoutubeRequest) -> Dict[str, str]:
    task = summarize_video_task.delay("youtube", req.url, req.ratio)
    return {"task_id": task.id}


@app.get("/tasks/{task_id}")
def task_status(task_id: str) -> Dict[str, Any]:
    result = AsyncResult(task_id, app=celery_app)
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "state": result.state,
        "meta": result.info if isinstance(result.info, dict) else {},
    }
    if result.state == "SUCCESS":
        payload["result"] = result.get()
    elif result.state in {"FAILURE", "REVOKED"}:
        payload["error"] = str(result.result)
    return payload


@app.get("/tasks/{task_id}/stream")
async def task_stream(task_id: str):
    """SSE endpoint — pushes progress events as the Celery task runs."""

    async def _generate():
        last_progress = -1
        last_state = None
        deadline = asyncio.get_event_loop().time() + 600  # 10-min hard timeout

        while asyncio.get_event_loop().time() < deadline:
            result = AsyncResult(task_id, app=celery_app)
            state = result.state
            meta = result.info if isinstance(result.info, dict) else {}
            progress = int(meta.get("progress", 0))
            stage = meta.get("stage", state)

            changed = state != last_state or progress != last_progress

            if state == "SUCCESS":
                payload = {"state": state, "progress": 100, "stage": "Done", "result": result.get()}
                yield f"data: {json.dumps(payload)}\n\n"
                return

            if state in {"FAILURE", "REVOKED"}:
                payload = {"state": state, "progress": 0, "stage": state, "error": str(result.result)}
                yield f"data: {json.dumps(payload)}\n\n"
                return

            if changed:
                payload = {"state": state, "progress": progress, "stage": stage}
                yield f"data: {json.dumps(payload)}\n\n"
                last_state = state
                last_progress = progress
            else:
                yield ": heartbeat\n\n"

            await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'state': 'TIMEOUT', 'error': 'Job timed out after 10 minutes'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/rag/query")
def rag_query(req: RagQueryRequest) -> Dict:
    if not is_indexed(req.video_hash):
        raise HTTPException(status_code=404, detail="Video not indexed. Run summarization first.")
    return query_video(req.video_hash, req.question, n_results=req.n_results)


@app.post("/agent/transcript-insights")
def transcript_insights(req: TranscriptInsightRequest) -> Dict[str, Optional[str]]:
    output = transcript_pipeline.invoke({"transcript": req.transcript})
    return {
        "summary": output.get("summary", ""),
        "mode": output.get("mode", "unknown"),
    }
