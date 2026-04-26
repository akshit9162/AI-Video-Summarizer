import os
import tempfile
from typing import Any, Dict, Optional

from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from celery_app import celery_app
from src.agent_pipeline import build_transcript_agent_pipeline
from tasks import summarize_video_task

app = FastAPI(title="Video Summarization API", version="0.1.0")
transcript_pipeline = build_transcript_agent_pipeline()


class SummarizeYoutubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    ratio: float = Field(0.2, ge=0.05, le=0.5, description="Summary ratio")


class TranscriptInsightRequest(BaseModel):
    transcript: str = Field(..., min_length=1)


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


@app.post("/agent/transcript-insights")
def transcript_insights(req: TranscriptInsightRequest) -> Dict[str, Optional[str]]:
    output = transcript_pipeline.invoke({"transcript": req.transcript})
    return {
        "summary": output.get("summary", ""),
        "mode": output.get("mode", "unknown"),
    }
