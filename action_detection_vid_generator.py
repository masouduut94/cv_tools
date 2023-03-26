import cv2
import pandas as pd
from tqdm import tqdm
from os.path import join
from pathlib import Path
from natsort import natsorted


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

    for i, clip in enumerate(all_clips):
        if i > quota:
            break
        frame_1st = clip[0]
        frame_last = clip[-1]
        name = join(save_path, filename + f'_{frame_1st}_{frame_last}.mp4')

        for frame in clip:
            cap.set(1, frame)
            status, frame = cap.read()
            writer = cv2.VideoWriter(name, codec, fps, (w, h))
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

    for vid, csv in tqdm(list(zip(train_videos, train_csvs))):
        df = pd.read_csv(csv.as_posix())
        get_serve_videos(df, train_serve, vid.as_posix())
        get_no_serves(df, train_noserve, vid.as_posix(), clip_length=40, quota=100)

    for vid, csv in tqdm(list(zip(test_videos, test_csvs))):
        df = pd.read_csv(csv.as_posix())
        get_serve_videos(df, test_serve, vid.as_posix())
        get_no_serves(df, test_noserve, vid.as_posix(), clip_length=40, quota=100)