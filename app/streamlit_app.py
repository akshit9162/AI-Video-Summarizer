import time

import requests
import streamlit as st

st.title("🎬 AI Video Summarizer")
API_BASE = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000").rstrip("/")

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
        source_type = "upload"
        source_value = f
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
    try:
        if source_type == "upload":
            files = {"file": (source_value.name, source_value.getvalue(), "video/mp4")}
            resp = requests.post(
                f"{API_BASE}/summarize/upload",
                files=files,
                params={"ratio": ratio / 100.0},
                timeout=120,
            )
        else:
            resp = requests.post(
                f"{API_BASE}/summarize/youtube",
                json={"url": source_value, "ratio": ratio / 100.0},
                timeout=30,
            )
        resp.raise_for_status()
        st.session_state.task_id = resp.json()["task_id"]
        st.session_state.task_result = None
    except requests.RequestException as exc:
        st.error(f"Failed to start job via FastAPI: {exc}")
        st.stop()

if st.session_state.task_id:
    try:
        status_resp = requests.get(f"{API_BASE}/tasks/{st.session_state.task_id}", timeout=20)
        status_resp.raise_for_status()
        data = status_resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch job status from FastAPI: {exc}")
        st.stop()

    state = data.get("state", "PENDING")
    meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
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
        res = data.get("result", {})
        st.session_state.task_result = res
        st.session_state.task_id = None
        status_text.success("Job complete")
    elif state in {"FAILURE", "REVOKED"}:
        st.error(f"Job failed: {data.get('error', 'Unknown error')}")
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