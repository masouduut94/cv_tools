"""
This script slices the videos based on:
- start time
- the amount of seconds to include after that event.

"""

import cv2
import numpy as np
import argparse

video_name = 'videos/rally4.mp4'

cap = cv2.VideoCapture(video_name)
w = int(cap.get(3))
h = int(cap.get(4))
fps = int(cap.get(5))
n_frames = int(cap.get(7))

st_input = input("start time of the split: ")
how_many_seconds = input("how many seconds to include: ")
filename = input("output video name: ")

st_input = int(st_input) * fps
how_many_seconds = int(how_many_seconds) * fps

cap.set(1, st_input)
codec = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(filename, codec, fps, (w, h))

for i in range(how_many_seconds):
	_, frame = cap.read()
	writer.write(frame)

writer.release()




# Give 2 argument
# 1st: start time to split
# 2nd: how many seconds after that.


def seconds2time(fps, frame_no):
    seconds = frame_no // fps
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    dt = f'{h:d}-{m:02d}-{s:02d}'
    return dt


