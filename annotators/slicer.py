import cv2
from os.path import join
from pathlib import Path

VIDEO_NAME = 'E:\\TVConal\\TableTennis\\codes\\data\\test_videos\\test1.mp4'
OUTPUT_PATH = "E:\\TVConal\\TableTennis\\codes\\data\\OUTPUT_VIDEOS"
cap = cv2.VideoCapture(VIDEO_NAME)
assert Path(VIDEO_NAME).is_file(), "does not exist..."
condition = True
while condition:
    key = cv2.waitKeyEx(1)
    if key == ord('q'):
        condition = True

    st_frame = int(input('START FRAME: '))
    total_frames = int(input('HOW MANY FRAMES: '))
    
    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]

    cap.set(1, st_frame)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output = join(OUTPUT_PATH, Path(VIDEO_NAME).stem + f'st_{st_frame}_end_{st_frame + total_frames}_output.mp4')
    writer = cv2.VideoWriter(output, codec, fps, (w, h))
    for i in range(st_frame, (st_frame+total_frames)):
        cap.set(1, i)
        status, frame = cap.read()
        writer.write(frame)
    writer.release()




