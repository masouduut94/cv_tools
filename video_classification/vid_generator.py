from os import makedirs

import cv2
import random as r
import pandas as pd
from tqdm import tqdm
from os.path import join
from pathlib import Path
from natsort import natsorted
# import torchvision.transforms as T
# from albumentations import HorizontalFlip
import albumentations as A


def get_serve_videos(df, save_path, filename, apply_flip=True):
    toss = df[df.toss].frame.tolist()
    end_toss = df[df.toss_end].frame.tolist()

    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()
    w, h, fps = [int(cap.get(i)) for i in range(3, 6)]
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    filename = Path(filename).stem

    st_end_pairs = list(zip(toss, end_toss))
    transform = A.Compose([A.HorizontalFlip(always_apply=True)])

    for (start_frame, end_frame) in tqdm(st_end_pairs, desc="processing serves"):
        name1 = join(save_path, filename + f'_{start_frame}_{end_frame}.mp4')
        writer1 = cv2.VideoWriter(name1, codec, fps, (w, h))

        for fno in range(start_frame, end_frame):
            cap.set(1, fno)
            status, frame = cap.read()
            writer1.write(frame)
            flipped_frame = transform(image=frame)["image"]
        writer1.release()

        if apply_flip:
            name2 = join(save_path, filename + f'_{start_frame}_{end_frame}_flipped.mp4')
            writer2 = cv2.VideoWriter(name2, codec, fps, (w, h))
            for fno in range(start_frame, end_frame):
                cap.set(1, fno)
                status, frame = cap.read()
                flipped_frame = transform(image=frame)["image"]
                writer2.write(flipped_frame)
            writer2.release()


def get_frames_in_slices(frames_list, clip_length):
    # Make sure we can create as many as videos as we can from the list of frames.
    div, mod = divmod(len(frames_list), clip_length)
    clips = []
    for i in range(div):
        item = frames_list[i * clip_length: clip_length * (i + 1)]
        clips.append(item)
    return clips


def get_no_serves(df, save_path, filename, clip_length=60, quota=100):
    all_frames = set(df.frame.tolist())

    start_toss = df[df.toss].frame.tolist()
    end_toss = df[df.toss_end].frame.tolist()

    exclude_frames = df[df.exclude].frame.tolist()
    end_exclude_frames = df[df.exclude_end].frame.tolist()

    remove_frames = []

    for i, j in zip(exclude_frames, end_exclude_frames):
        t = list(range(i, j))
        remove_frames += t

    for i, j in zip(start_toss, end_toss):
        t = list(range(i, j))
        remove_frames += t

    all_frames.difference_update(remove_frames)

    slices = []
    temp = []

    all_frames = list(all_frames)
    for i, _ in enumerate(all_frames[:-1]):
        if abs(all_frames[i] - all_frames[i + 1]) == 1:
            temp.append(all_frames[i])
        else:
            if len(temp) < clip_length:
                temp = []
                continue
            else:
                slices.append(temp)
                temp = []

    all_clips = []
    for slicee in slices:
        clips = get_frames_in_slices(slicee, clip_length)
        all_clips += clips

    cap = cv2.VideoCapture(filename)
    assert cap.isOpened(), f"file is not accessible {filename}"

    w, h, fps = [int(cap.get(i)) for i in range(3, 6)]
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    filename = Path(filename).stem

    all_clips = r.choices(all_clips, k=quota) if quota < len(all_clips) else all_clips

    for i, clip in enumerate(tqdm(all_clips, desc='processing no_serves')):
        frame_1st = clip[0]
        frame_last = clip[-1]
        name = join(save_path, filename + f'_{frame_1st}_{frame_last}.mp4')
        writer = cv2.VideoWriter(name, codec, fps, (w, h))
        for frame in clip:
            cap.set(1, frame)
            status, frame = cap.read()
            writer.write(frame)
        writer.release()


if __name__ == '__main__':

    # Generate Train data
    train_videos = natsorted(list(Path('../videos/train').glob('*.mp4')), key=lambda x: x.as_posix())
    train_csvs = natsorted(list(Path('../labels/train').glob('*.csv')), key=lambda x: x.as_posix())
    train_serve = 'test_algo/train/serve'
    train_noserve = 'test_algo/train/no_serve'

    makedirs(train_serve, exist_ok=True)
    makedirs(train_noserve, exist_ok=True)

    for vid, csv in tqdm(list(zip(train_videos[1:2], train_csvs[1:2]))):
        df = pd.read_csv(csv.as_posix())
        get_serve_videos(df, train_serve, vid.as_posix(), apply_flip=True)
        get_no_serves(df, train_noserve, vid.as_posix(), clip_length=30, quota=20)

    # Generate test data
    test_videos = natsorted(list(Path('../videos/test').glob('*.mp4')), key=lambda x: x.as_posix())
    test_csvs = natsorted(list(Path('../labels/test').glob('*.csv')), key=lambda x: x.as_posix())
    test_serve = 'data/test/serve'
    test_noserve = 'data/test/no_serve'

    makedirs(test_serve, exist_ok=True)
    makedirs(test_noserve, exist_ok=True)

    for vid, csv in tqdm(list(zip(test_videos, test_csvs))):
        df = pd.read_csv(csv.as_posix())
        get_serve_videos(df, test_serve, vid.as_posix(), apply_flip=False)
        get_no_serves(df, test_noserve, vid.as_posix(), clip_length=40, quota=60)
