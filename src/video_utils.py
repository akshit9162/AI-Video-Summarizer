import cv2, numpy as np, torch, clip
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(path, n=80):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        raise ValueError("❌ Video has 0 frames")

    idxs = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret and frame is not None:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("❌ No frames extracted (video decode issue)")

    return np.array(frames)

def extract_features(frames):
    valid_frames = [f for f in frames if f is not None]

    if len(valid_frames) == 0:
        raise ValueError("❌ No valid frames for feature extraction")

    imgs = torch.stack([
        preprocess(Image.fromarray(f)) for f in valid_frames
    ])

    with torch.no_grad():
        feats = model.encode_image(imgs)

    return feats.numpy()

def importance(feats):
    m = feats.mean(0)
    d = np.linalg.norm(feats-m, axis=1)
    return (d-d.min())/(d.max()-d.min()+1e-8)

def video_info(p):
    cap = cv2.VideoCapture(p)
    fps = cap.get(5)
    total = int(cap.get(7))
    cap.release()
    return fps, total, total/fps

def map_frames(total, n):
    return np.linspace(0, total-1, n, dtype=int)