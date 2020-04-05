import cv2
import skvideo.io as vio
import numpy as np


def read_all_video_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    video_frames = []
    while cap.isOpened():
        has_frame, frame = cap.read()
        if has_frame:
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    cap.release()
    return video_frames


def get_video_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25
    cap.release()
    return fps


def read_video_frames(video_file, desired_fps):
    fps = get_video_fps(video_file)
    assert desired_fps <= fps, 'the video {} has {} fps'.format(video_file, fps)
    cap = cv2.VideoCapture(video_file)

    sample_period = round(fps / desired_fps)
    video_frames = []
    n_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if int(n_frames % sample_period) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(frame)
        else:
            break
        n_frames += 1
    cap.release()

    # fallback method to load videos
    if len(video_frames) == 0:  # try loading from skvideo instead
        n_frames = 0
        in_params = {'-vcodec': 'h264'}
        reader = vio.FFmpegReader(video_file, inputdict=in_params)
        for frame in reader.nextFrame():
            if isinstance(frame, np.ndarray):
                if int(n_frames % sample_period) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_frames.append(frame)
            else:
                break
            n_frames += 1
        reader.close()

    assert len(video_frames) > 0, 'failed to read {}'.format(video_file)
    return video_frames
