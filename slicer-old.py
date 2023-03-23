import cv2
from os.path import join, isdir, isfile
from pathlib import Path
from os import makedirs
from tqdm import tqdm

INNING = 2


def seconds2time(fps, frame_no):
    seconds = frame_no // fps
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    dt = f'{h:d}-{m:02d}-{s:02d}'
    return dt


input_path = '/mnt/disk1/prod_data/mathew/'
text_files = Path('./').glob('*.txt')
codec = cv2.VideoWriter_fourcc(*'mp4v')

for txt in text_files:
    lines = [int(l) for l in txt.read_text().split('\n') if l.isnumeric()]
    lines = sorted(lines)
    video_file = join(input_path, txt.stem + '.MP4')
    cap = cv2.VideoCapture(video_file)

    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5))

    assert cap.isOpened(), f" video file not found {video_file}"
    output_path = join(f'inning2_highlights/{txt.stem}')
    makedirs(output_path, exist_ok=True)

    for i, frame_no in enumerate(tqdm(lines), 1):
        cap.set(1, frame_no-10)
        writer = cv2.VideoWriter(
            join(output_path, f'inning{INNING}_clip_{i}_time_{seconds2time(fps, frame_no)}.mp4'),
            codec, fps, (w, h))

        for _ in range(300):
            status, frame = cap.read()
            writer.write(frame)
