from typing import Union, List, Tuple

import cv2
from pathlib import Path
from os.path import join
import numpy as np
import pandas as pd
from shot_type_utils import *

"""
Data must be like this:

'shot_type': 'push', 'lob', 'loop', 'flick'
'hand_type': 'back-hand' 'fore-hand'
'player': 'left_player' - 'right_player'

frame | x | y | back-hand/fore-hand/serve | push/lob/loop/flick
"""

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global data, cap, current
    global frame

    if event == cv2.EVENT_LBUTTONDOWN:
        df.at[current, 'x'] = x
        df.at[current, 'y'] = y
        frame = to_frame(cap, df, current, n_frames)

    if event == cv2.EVENT_MOUSEWHEEL:
        print(flags)
        if flags == 7864320:  # MOUSEWHEEL UP
            current += SKIP1
        elif flags == -7864320:  # MOUSEWHEEL DOWN
            current -= SKIP1
        elif flags == 7864328:  # CTRL + MOUSEWHEEL UP
            current += SKIP2
        elif flags == -7864312:  # CTRL + MOUSEWHEEL DOWN
            current -= SKIP2
        elif flags == 7864336:  # SHIFT + MOUSEWHEEL UP
            current += SKIP3
        elif flags == -7864304:  # SHIFT + MOUSEWHEEL DOWN
            current -= SKIP3
        # current = (current + SKIP_WHEEL) if flags > 0 else (current - SKIP_WHEEL)
        frame = to_frame(cap, df, current, save_path=save_path, total_frame=n_frames)


if __name__ == '__main__':
    # CHANGE THIS FOR NEW WORK
    # MAKE SURE YOUR CSV FILE IS SAVED EACH TIME YOU ANNOTATE
    VIDEO_FILE = "E:\\TVConal\\TableTennis\\codes\\data\\annotated\\videos\\train\\cc.mp4"
    CSV_SAVE_PATH = 'E:\\TVConal\\TableTennis\\codes\\data\\annotated\\shot_types\\train'

    # CHANGE THIS FOR NEW WORK
    # if left_player == 1 => forehand - if == 2 => backhand
    # if right_player == 1 => forehand | - if == 2 => backhand

    all_dicts = {
        "shot_type":
            {
                'none': -1,
                'serve': 0,
                'push': 1,
                'flat': 2,
                'drive(loop)': 3,
                'flick': 4,
            },
        "hand_type":
            {
                'none': -1,
                'backhand': 0,
                'forehand': 1
            },
        "player":
            {
                'none': -1,
                'left': 0,
                'right': 1
            }
    }

    cols_dtype = {
        'int': ['player', 'hand_type', 'shot_type'],
        'bool': ['exclude', 'exclude_end']
    }

    msg_cols = ['player', 'hand_type', 'shot_type', 'exclude', 'exclude_end']

    cap = cv2.VideoCapture(VIDEO_FILE)
    assert cap.isOpened(), "file is not opened!"
    name = Path(VIDEO_FILE).stem

    save_path = join(CSV_SAVE_PATH, name + '.csv')

    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]
    current = 0

    try:
        df = pd.read_csv(save_path)
        df = init(df, cols_dtype, n_frames=n_frames)
        column = 'player'
        next_frame, msg = go_to_next(df, column=column, value=(0, 1), current=current, return_last=True)
        if next_frame is not None:
            frame = to_frame(cap, df, next_frame, n_frames, custom_msg=f'next {column}')
        print(f"loading from csv file {save_path}")
    except:
        df = init(None, cols_dtype, n_frames=n_frames, with_fake_values=True)
        print(f"failed to load {save_path}. initializing ......")

    frame = to_frame(cap, df, current, n_frames, save_path=save_path)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        # frame = cv2.resize(frame, (w, h))
        cv2.imshow("image", frame)
        key = cv2.waitKeyEx(1)  # & 0xFF
        if key != -1:
            print(key)

        if key == 27:  # Esc
            df = save_data(df, save_path)
            custom_msg = "Data is saved ..."
            frame = to_frame(cap, df, current, n_frames, custom_msg=custom_msg, save_path=save_path)
            break

        elif key == ord('1'):
            # Toggle between player values
            # 1, 2
            col = 'player'
            output = ''
            _ = toggle(df, current, col, all_dicts[col])
            frame = to_frame(cap, df, current, n_frames, save=True, save_path=save_path)

        elif key == ord('2'):
            # Toggle between values 0, 1, 2
            col = 'hand_type'
            _ = toggle(df, current, col, all_dicts[col])
            frame = to_frame(cap, df, current, n_frames, save=True, save_path=save_path)

        elif key == ord('3'):
            # Toggle between values 0, 1, 2
            col = 'shot_type'
            output = 'shot: '
            _ = toggle(df, current, col, all_dicts[col])
            frame = to_frame(cap, df, current, n_frames, save=True, save_path=save_path)

        elif key == ord('4'):
            # CHANGE (toss_end) FOR NEW WORK
            col = 'exclude'
            df.at[current, col] = False if df.at[current, col] else True
            frame = to_frame(cap, df, current, n_frames, save=True, save_path=save_path)
            print(current, f" {col}: {df.at[current, col]}")

        elif key == ord('5'):
            # CHANGE (toss_end) FOR NEW WORK
            col = 'exclude_end'
            df.at[current, col] = False if df.at[current, col] else True
            frame = to_frame(cap, df, current, n_frames, save=True, save_path=save_path)
            print(current, f" {col}: {df.at[current, col]}")

        elif key == ord('s'):
            df = save_data(df, save_path)
            custom_msg = "Data is saved ..."
            frame = to_frame(cap, df, current, n_frames, save_path=save_path, custom_msg=custom_msg)
        elif key == ord("f"):
            try:
                check = int(input('Enter your frame:'))
            except:
                print("not a valid number.")
                check = current
            frame = to_frame(cap, df, check, n_frames, save_path=save_path)
        elif key == ord('x'):
            # Skip to next frame
            check = current + 1
            frame = to_frame(cap, df, check, n_frames, save_path=save_path)
        elif key == ord('z'):
            # Skip to previous frame
            check = current - 1
            frame = to_frame(cap, df, check, n_frames, save_path=save_path)

        elif key == 2555904:
            # Go to next unlabeled value (Right Arrow =>)
            column = 'player'
            next_frame, msg = go_to_next(df, column=column, value=(0, 1), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,  custom_msg=f'jumping next {column}')
        elif key == 2424832:
            column = 'player'
            # Go to next unlabeled value (Left Arrow <=)
            next_frame, msg = go_to_previous(df, column=column, value=(0, 1), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,  custom_msg=f'jumping previous {column}')
        elif key == 2490368:
            # Go to next unlabeled value (Up Arrow <=)
            column = 'shot_type'
            next_frame, msg = go_to_next(df, column=column, value=tuple(range(0, 6)), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,  custom_msg=f'jumping next {column}')
        elif key == 2621440:
            # Go to previous unlabeled value (Bottom Arrow)
            column = 'shot_type'
            next_frame, msg = go_to_previous(df, column=column, value=tuple(range(0, 5)), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path, custom_msg=f'jumping previous {column}')

