# AI Video Summarization using Reinforcement Learning

## 1) Project Overview

This project is an end-to-end AI system that generates short highlight videos from long-form input videos (uploaded files or YouTube URLs). The core summarization logic uses a reinforcement learning (RL) agent that selects informative and diverse frames, then reconstructs a condensed video while preserving audio-video coherence. A Streamlit frontend and Celery + Redis backend provide an asynchronous user experience suitable for real-world usage.

## 2) Problem Statement

Long videos are expensive to consume and difficult to browse. The goal is to automatically create concise summaries that:

- preserve important moments,
- maintain content diversity,
- avoid redundant frames,
- and return a playable summarized video with minimal manual editing.

## 3) Key Contributions

- Built a two-stage RL policy architecture (horizontal + vertical actions) for frame-level summarization decisions.
- Designed a custom video summarization environment with reward shaping for importance, diversity, and coverage.
- Integrated scene-aware video reconstruction with fallback logic for robustness.
- Added asynchronous task orchestration with Celery/Redis for non-blocking web UX.
- Implemented speech transcript extraction and caching to enrich output beyond visual highlights.
- Added practical production safeguards for macOS worker stability (solo pool, controlled concurrency).

## 4) System Architecture

1. **Input Layer**
   - User uploads an MP4 or provides a YouTube URL in Streamlit.
   - YouTube videos are fetched with `yt-dlp` in a Celery task.

2. **Feature Extraction Layer**
   - Frames are sampled from the source video.
   - CLIP image embeddings are computed for semantic representation.
   - Frame importance is estimated from embedding distance statistics.

3. **RL Summarization Layer**
   - The environment maintains a set of selected frame indices.
   - **Horizontal policy** chooses which selected index to update.
   - **Vertical policy** chooses movement action (`-1`, `+1`, `-5`, `+5`) to refine that index.
   - Reward combines:
     - mean importance of selected frames,
     - diversity among selected features,
     - temporal coverage across the video.

4. **Video Reconstruction Layer**
   - Scene boundaries are detected using PySceneDetect.
   - Selected frame anchors are mapped back to scene intervals.
   - Summary video is written with OpenCV and merged with original audio via FFmpeg.

5. **Speech Layer**
   - Transcript is generated with Faster-Whisper (`tiny.en`) and cached by file hash.
   - Cached transcript is surfaced in the UI for quick consumption.

6. **Serving Layer**
   - Streamlit provides interactive UI and progress bars.
   - Celery workers execute long-running processing jobs asynchronously.
   - Redis serves as broker/result backend for task state and progress updates.

## 5) Technology Stack

- **Language:** Python
- **ML/DL:** PyTorch, CLIP, Faster-Whisper
- **Video Processing:** OpenCV, FFmpeg, PySceneDetect
- **Backend Orchestration:** Celery, Redis
- **Frontend:** Streamlit
- **Data/Utils:** NumPy, Pillow, yt-dlp

## 6) Core Algorithms and Design

### A) RL Formulation

- **State:** embeddings of currently selected key frames.
- **Action space:**
  - Horizontal action: select one candidate index among `K`.
  - Vertical action: shift selected frame index in temporal space.
- **Reward design:**
  - Importance term encourages salient moments.
  - Diversity term discourages repetitive frames.
  - Coverage term promotes temporal spread.
- **Training objective:** policy-gradient style optimization with entropy regularization for exploration.

### B) Inference Strategy

- Extract visual features from sampled frames.
- Run deterministic policy rollout for stable summaries.
- Apply scene constraints to improve coherence of generated highlights.
- Merge audio from original stream with generated summary video track.

## 7) Reliability and Engineering Decisions

- Asynchronous processing prevents UI freeze during long jobs.
- Progress callback reporting improves user visibility during download/inference.
- Hash-based transcript caching avoids repeated transcription cost.
- Fallback write path ensures output generation even when scene mapping is sparse.
- Worker configuration is tuned for macOS stability where heavy native libs may crash under prefork multiprocessing.

## 8) Results and Current Output Quality

The system produces:

- a playable summarized MP4,
- a highlight visualization plot,
- and a cached speech transcript.

Internal quality signals include:

- mean selected-frame importance,
- temporal coverage of selected indices.

These metrics are useful for comparing checkpoints and tuning summarization behavior.

## 9) Challenges Faced and How They Were Solved

- **Long-running inference in UI:** solved with Celery + Redis async execution.
- **Platform-specific worker instability:** mitigated via single-process (`solo`) worker strategy.
- **Scene/selection mismatches:** handled with fallback frame writing to guarantee output.
- **Repeated transcript cost:** solved by content-hash cache for transcript reuse.

## 10) Resume-Ready Impact Bullets

Use these directly in your resume (edit numbers once you benchmark on your machine):

- Built an end-to-end **AI video summarization pipeline** combining RL-based keyframe selection, scene detection, and audio-preserving video synthesis.
- Designed a **custom RL environment** with multi-objective reward shaping (importance, diversity, coverage) for improved highlight quality.
- Productionized inference with **Celery + Redis asynchronous architecture**, enabling non-blocking UX with real-time progress reporting in Streamlit.
- Integrated **speech transcription with caching**, reducing repeated processing overhead and improving output explainability.
- Engineered a robust media pipeline with **OpenCV + FFmpeg**, including fallback logic to guarantee summary generation under edge cases.

## 11) Interview Talking Points

- Why hierarchical actions (horizontal + vertical) are useful for large temporal search spaces.
- How reward shaping influences summary style and trade-offs.
- Why async architecture is essential for multimedia ML workloads.
- How you balanced model complexity with practical reliability on local hardware.
- What you would measure next: compression ratio, user preference scores, and latency by stage.

## 12) Future Improvements

- Add timestamped transcript chunks and semantic Q&A over transcript (RAG).
- Introduce optional fast mode (lower frame sampling, optional transcript stage).
- Replace/reinforce handcrafted importance signal with learned saliency head.
- Add benchmark suite for latency/quality across video categories.
- Package services with Docker and add CI checks for reproducible deployment.

## 13) One-Line Resume Summary

Developed a full-stack AI video summarization system that uses reinforcement learning, scene-aware reconstruction, and asynchronous task orchestration to generate concise, audio-preserving highlights from long videos.

