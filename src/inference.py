import os
import shutil
import torch
from src.video_utils import *
from src.env import VideoSummarizationEnv
from src.model import *
from src.scene_detection import detect_scenes
from src.visualize import create_summary
from src.highlight_vis import plot_highlights
from src.evaluate import evaluate
from src.speech_summary import speech_summary

FFMPEG = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
DEFAULT_CHECKPOINT = "checkpoints/policies.pt"


def _load_policies(feature_dim, k, checkpoint_path=None):
    hp = HorizontalPolicy(feature_dim, K=k)
    vp = VerticalPolicy(feature_dim)

    ckpt_path = checkpoint_path or DEFAULT_CHECKPOINT
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hp.load_state_dict(ckpt["horizontal_policy"])
        vp.load_state_dict(ckpt["vertical_policy"])
        print(f"Loaded policy checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}. Using untrained policies.")

    hp.eval()
    vp.eval()
    return hp, vp


def run_inference(
    video,
    output,
    target_ratio=0.2,
    checkpoint_path=None,
    deterministic=True,
    progress_callback=None,
):
    def _progress(value, message):
        if progress_callback is not None:
            progress_callback(float(value), message)

    if FFMPEG is None:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or set FFMPEG_PATH environment variable."
        )

    _progress(0.05, "Extracting frames")
    frames = extract_frames(video)
    _progress(0.20, "Extracting CLIP features")
    feats = extract_features(frames)
    imp = importance(feats)

    env = VideoSummarizationEnv(feats, imp)
    state = torch.tensor(env.reset(), dtype=torch.float32)

    _progress(0.35, "Loading RL policies")
    hp, vp = _load_policies(feats.shape[1], env.K, checkpoint_path=checkpoint_path)

    _progress(0.40, "Running RL summarization policy")
    num_steps = 40
    for step in range(num_steps):
        h_probs = hp(state)
        if deterministic:
            ah = torch.argmax(h_probs).item()
        else:
            ah = torch.multinomial(h_probs, 1).item()

        v_probs = vp(state, ah)
        if deterministic:
            av = torch.argmax(v_probs).item()
        else:
            av = torch.multinomial(v_probs, 1).item()

        state,_ ,_= env.step(ah,av)
        state = torch.tensor(state,dtype=torch.float32)
        _progress(0.40 + 0.30 * ((step + 1) / num_steps), "Running RL summarization policy")

    fps,total,dur = video_info(video)
    maxf = max(1, int(dur * target_ratio * fps))

    _progress(0.75, "Detecting scenes")
    scenes = detect_scenes(video)
    fmap = map_frames(total,len(frames))

    temp="temp.mp4"
    _progress(0.85, "Creating summary video")
    written = create_summary(video, fmap, env.selected_idx, scenes, temp, fps, maxf)
    if written == 0:
        raise RuntimeError("Failed to generate summary frames from the input video.")

    _progress(0.92, "Merging audio and video")
    ffmpeg_cmd = (
        f'"{FFMPEG}" -y -i "{video}" -i "{temp}" '
        f'-map 0:a -map 1:v -shortest "{output}"'
    )
    if os.system(ffmpeg_cmd) != 0:
        raise RuntimeError("ffmpeg merge failed while creating summary output.")

    _progress(0.96, "Generating speech summary")
    speech = speech_summary(video)
    print("Metrics:", evaluate(feats, imp, env.selected_idx))

    _progress(0.99, "Preparing highlights")
    plot_path = plot_highlights(imp, env.selected_idx)
    _progress(1.0, "Done")

    return {"video": output, "plot": plot_path, "speech_summary": speech}