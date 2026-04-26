import os
import sys
import tempfile
import time

from celery.result import AsyncResult
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from celery_app import celery_app
from tasks import summarize_video_task

st.title("🎬 AI Video Summarizer")

if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "task_result" not in st.session_state:
    st.session_state.task_result = None

opt = st.radio("Input:", ["Upload", "YouTube"])
source_type = None
source_value = None

if opt == "Upload":
    f = st.file_uploader("Upload", type=["mp4"])
    if f:
        os.makedirs("uploads", exist_ok=True)
        upload_path = os.path.join("uploads", f"{next(tempfile._get_candidate_names())}.mp4")
        with open(upload_path, "wb") as fp:
            fp.write(f.read())
        source_type = "upload"
        source_value = upload_path
else:
    url = st.text_input("YouTube URL")
    if url:
        source_type = "youtube"
        source_value = url

ratio = st.slider("Summary %", 5, 50, 20)

download_bar = st.progress(0, text="Download progress")
summarize_bar = st.progress(0, text="Summarization progress")
status_text = st.empty()

if st.button("Run"):
    if not source_type or not source_value:
        st.error("Please provide a valid input video source.")
        st.stop()
    task = summarize_video_task.delay(source_type, source_value, ratio / 100.0)
    st.session_state.task_id = task.id
    st.session_state.task_result = None

if st.session_state.task_id:
    task_result = AsyncResult(st.session_state.task_id, app=celery_app)
    state = task_result.state
    meta = task_result.info if isinstance(task_result.info, dict) else {}
    progress = int(meta.get("progress", 0))
    stage = meta.get("stage", state)

    if progress <= 35:
        download_bar.progress(progress, text=f"Download progress: {progress}%")
        summarize_bar.progress(0, text="Summarization progress: waiting")
    else:
        download_bar.progress(100, text="Download progress: complete")
        summarize_pct = int((progress - 35) / 65 * 100)
        summarize_bar.progress(
            max(0, min(100, summarize_pct)),
            text=f"Summarization progress: {stage}",
        )

    status_text.info(f"Job status: {state} | {stage}")

    if state == "SUCCESS":
        res = task_result.get()
        st.session_state.task_result = res
        st.session_state.task_id = None
        status_text.success("Job complete")
    elif state in {"FAILURE", "REVOKED"}:
        st.error(f"Job failed: {task_result.result}")
        st.session_state.task_id = None
    else:
        time.sleep(2)
        st.rerun()

if st.session_state.task_result:
    res = st.session_state.task_result
    st.video(res["video"])
    st.image(res["plot"])
    if res.get("speech_transcript"):
        st.subheader("Speech Transcript")
        st.write(res["speech_transcript"])