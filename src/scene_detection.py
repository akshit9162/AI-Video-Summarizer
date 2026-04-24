from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path):
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector())

    vm.start()
    sm.detect_scenes(frame_source=vm)

    scenes = sm.get_scene_list()
    vm.release()

    return [(s[0].get_frames(), s[1].get_frames()) for s in scenes]