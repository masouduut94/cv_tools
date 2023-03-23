from pathlib import Path, PosixPath
import os
from os.path import join, isdir
from os import makedirs
import pandas as pd
from tqdm import tqdm
import cv2
from natsort import natsorted
import random as r


def get_serve_videos(df, save_path, filename):
    toss = df[df.toss].frame.tolist()
    end_toss = df[df.toss_end].frame.tolist()

    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()
    w, h, fps = [int(cap.get(i)) for i in range(3, 6)]
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    filename = Path(filename).stem

    st_end_pairs = list(zip(toss, end_toss))

    for (start_frame, end_frame) in tqdm(st_end_pairs):
        name = join(save_path, filename + f'_{start_frame}_{end_frame}.mp4')
        writer = cv2.VideoWriter(name, codec, fps, (w, h))
        for fno in range(start_frame, end_frame):
            cap.set(1, fno)
            status, frame = cap.read()
            writer.write(frame)
        writer.release()


def get_no_serves(df, save_path, filename, quota=30, clip_length=60):
    start_toss = df[df.toss].frame.tolist()
    end_toss = df[df.toss_end].frame.tolist()
    _ = start_toss.pop(0)

    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()

    w, h, fps = [int(cap.get(i)) for i in range(3, 6)]
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    filename = Path(filename).stem

    start_end_pairs = list(zip(end_toss, start_toss))

    # for start, end in start_end_pairs:
    #     times = (start - end) // quota
    #     min_frames = (start - end) % quota

    if quota < len(start_end_pairs):
        choices = r.choices(start_end_pairs, k=quota)
    else:
        choices = start_end_pairs

    for (start_frame, end_frame) in tqdm(choices):
        # print(start_frame, end_frame - clip_length)
        if (end_frame - clip_length) - start_frame < 20:
            print("low frames included...")
            continue

        rnd_start = r.randint(start_frame, end_frame - clip_length)
        end = rnd_start + clip_length

        name = join(save_path, filename + f'_{start_frame}_{end_frame}.mp4')
        writer = cv2.VideoWriter(name, codec, fps, (w, h))
        for fno in range(rnd_start, end):
            cap.set(1, fno)
            status, frame = cap.read()
            writer.write(frame)
        writer.release()


if __name__ == '__main__':
    train_videos = natsorted(list(Path('videos/train').glob('*.mp4')), key=lambda x: x.as_posix())
    test_videos = natsorted(list(Path('videos/test').glob('*.mp4')), key=lambda x: x.as_posix())
    train_csvs = natsorted(list(Path('labels/train').glob('*.csv')), key=lambda x: x.as_posix())
    test_csvs = natsorted(list(Path('labels/test').glob('*.csv')), key=lambda x: x.as_posix())

    train_serve = 'data/train/serve'
    train_noserve = 'data/train/no_serve'

    test_serve = 'data/test/serve'
    test_noserve = 'data/test/no_serve'

    for vid, csv in zip(train_videos[1:], train_csvs[1:]):
        df = pd.read_csv(csv.as_posix())
        get_serve_videos(df, train_serve, vid.as_posix())
        get_no_serves(df, train_noserve, vid.as_posix(), quota=140, clip_length=40)

    for vid, csv in zip(test_videos, test_csvs):
        df = pd.read_csv(csv.as_posix())
        get_serve_videos(df, test_serve, vid.as_posix())
        get_no_serves(df, test_noserve, vid.as_posix(), quota=140, clip_length=40)




