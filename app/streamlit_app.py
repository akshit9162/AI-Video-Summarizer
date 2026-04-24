import glob
import os
import sys
import tempfile

import streamlit as st
import yt_dlp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import run_inference

st.title("🎬 AI Video Summarizer")

opt = st.radio("Input:", ["Upload","YouTube"])
path=None

if opt=="Upload":
    f=st.file_uploader("Upload",type=["mp4"])
    if f:
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        tmp.write(f.read()); path=tmp.name
else:
    url=st.text_input("YouTube URL")
    path = None

ratio=st.slider("Summary %",5,50,20)

download_bar = st.progress(0, text="Download progress")
summarize_bar = st.progress(0, text="Summarization progress")

if st.button("Run"):
    try:
        if opt == "Upload":
            if not path:
                st.error("Please upload an mp4 file first.")
                st.stop()
            download_bar.progress(100, text="Download progress: upload already available")
            source_path = path
        else:
            if not url:
                st.error("Please enter a YouTube URL.")
                st.stop()

            tmp_dir = tempfile.mkdtemp()
            out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

            def _download_progress_hook(d):
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes", 0)
                    if total > 0:
                        pct = int(min(100, (downloaded / total) * 100))
                        download_bar.progress(pct, text=f"Download progress: {pct}%")
                elif d.get("status") == "finished":
                    download_bar.progress(100, text="Download progress: complete")

            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "outtmpl": out_tmpl,
                "noplaylist": True,
                "quiet": True,
                "progress_hooks": [_download_progress_hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            candidates = sorted(glob.glob(os.path.join(tmp_dir, "*.mp4")))
            if not candidates:
                st.error("Failed to download YouTube video as mp4.")
                st.stop()
            source_path = candidates[-1]

        out=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")

        def _summarize_progress(value, message):
            pct = int(min(100, max(0, value * 100)))
            summarize_bar.progress(pct, text=f"Summarization progress: {message} ({pct}%)")

        res=run_inference(source_path,out.name,ratio/100,progress_callback=_summarize_progress)
        summarize_bar.progress(100, text="Summarization progress: complete")

        st.video(res["video"])
        st.image(res["plot"])
        if res.get("speech_summary"):
            st.subheader("Speech Summary")
            st.write(res["speech_summary"])
    except Exception as e:
        st.error(f"Summarization failed: {e}")