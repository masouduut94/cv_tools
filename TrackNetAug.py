from os import makedirs

import cv2
import pandas as pd
from pathlib import Path
from os.path import join
from natsort import natsorted
from tqdm import tqdm


def flipHorizontalVideo(vdo: cv2.VideoCapture, output_file):
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    W = int(vdo.get(3))
    H = int(vdo.get(4))
    FPS = int(vdo.get(5))
    size = (W, H)

    writer = cv2.VideoWriter(output_file, codec, FPS, size)
    status, frame = vdo.read()
    while status:
        frame = cv2.flip(frame, 1)
        writer.write(frame)
        status, frame = vdo.read()


def flipHorizontalCSV(csv_file: str, img_width, filename):
    df = pd.read_csv(csv_file)
    for i, _ in df.iterrows():
        if df.at[i, 'Visibility'] == 0:
            continue
        df.at[i, 'X'] = abs(img_width - df.at[i, 'X'])

    df.to_csv(filename, index=False)


if __name__ == '__main__':
    mp4s = list(Path('videos/NEW_ANNOTATED').rglob('*.mp4'))
    movs = list(Path('videos/NEW_ANNOTATED').rglob('*.mov'))
    all_videos = mp4s + movs
    all_csvs = list(Path('videos').rglob('*.csv'))

    vids = natsorted(all_videos, key=lambda x: (x.parent, x.stem))
    csvs = natsorted(all_csvs, key=lambda x: (x.parent, x.stem))

    zipped = list(zip(vids, csvs))

    for v, c in tqdm(zipped):
        parent = v.parent
        output = join('flipped', parent)
        makedirs(output, exist_ok=True)

        video_output_file = join(output, v.stem + '_flipped.mp4')
        csv_output_file = join(output, v.stem + '_flipped.csv')
        video = cv2.VideoCapture(v.as_posix())
        W = int(video.get(3))
        print("saved in ", video_output_file)
        flipHorizontalVideo(video, video_output_file)
        flipHorizontalCSV(c.as_posix(), W, csv_output_file)
