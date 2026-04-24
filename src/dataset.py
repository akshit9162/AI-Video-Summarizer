import os

def load_videos(folder):
    videos = []
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            videos.append(os.path.join(folder, file))
    return videos