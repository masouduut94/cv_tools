from typing import Union, List, Tuple

import cv2
from pathlib import Path
from os.path import join
import numpy as np
import pandas as pd

"""
Data must be like this:
frame | x | y | toss | end_toss | exclude | exclude-end
"""
SKIP1 = 1
SKIP2 = 10
SKIP_WHEEL = 15
SKIP3 = 200


def check_fno(fno, total_frame):
    """
    check if suggested frame number is not invalid based on video number of frames.
    Args:
        fno:
        total_frame:

    Returns:

    """
    if fno < 0:
        print('\nInvaild !!! Jump to first image...')
        return False
    elif fno > total_frame:
        print(f"\n maximum frames = {total_frame}")
    else:
        print(f"Frame set to: {fno}")
        return True


def to_frame(cap, df, current_fno, total_frame, save=False, custom_msg=None):
    global current
    print('current frame: ', current_fno)
    if check_fno(current_fno, total_frame):
        current = current_fno
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    ret, frame = cap.read()
    if save:
        save_data(df, save_path)
        print("The data is saved")
    if not (current >= total_frame):
        message = init_message(df, current, msg_cols, custom_msg)
        print("MESSAGE:   ", message)
    else:
        df = save_data(df, save_path)
        print("frame index bigger than number of frames.")
    if not ret:
        return None
    else:
        cv2.putText(frame, f'Frame: {current}/{total_frame}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if message != '':
            cv2.putText(frame, message, (100, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        item = df.iloc[current]
        if item.x != -1:
            color = (0, 0, 255)
            x = item.x
            y = item.y
            cv2.circle(frame, (x, y), 5, color, thickness=-1)
        return frame


def go_to_next(df: pd.DataFrame, column: str, value: Tuple[int, int], current: int, return_last=False):
    next_indexes = df.index[df.frame > current]
    msg = f"no more `{column}` after current value"
    if not len(next_indexes):
        print(msg)
        return None, msg

    temp = df[((df[column] == value[0]) | (df[column] == value[1])) & (df.index.isin(next_indexes))]
    if len(temp):
        if return_last:
            return temp.iloc[-1].frame, ""
        return temp.iloc[0].frame, ""
    else:
        print(msg)
        return None, msg


def go_to_previous(df: pd.DataFrame, column: str, value: Tuple[int, int], current: int, return_first=False):
    next_indexes = df.index[df.frame < current]
    msg = f"no more `{column}` before current value"
    if not len(next_indexes):
        print(msg)
        return None, msg

    temp = df[((df[column] == value[0]) | (df[column] == value[1])) & (df.index.isin(next_indexes))]
    if len(temp):
        if return_first:
            return temp.iloc[0].frame, ""
        return temp.iloc[-1].frame, ""
    else:
        print(msg)
        return None, msg


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
        frame = to_frame(cap, df, current, n_frames)


def init_message(df, index, columns, custom_msg=None):
    # data = df.iloc[index]
    st = ''

    for col in columns:
        if str(df.dtypes[col]).startswith('int'):
            val = df.at[index, col]
            if val <= 0:
                continue
            elif val == 1:
                st += f'{col} : forehand |'
            elif val == 2:
                st += f'{col} : backhand |'
        elif str(df.dtypes[col]).startswith('bool'):
            val = df.at[index, col]
            if val:
                st += f'{col} flag |'

    st = custom_msg if custom_msg is not None else st
    print(st)

    return st


def init(df: pd.DataFrame, cols_dtype: dict, with_fake_values: bool = False):
    """
    cols_dtype = {
        'int': ['right_player', 'left_player', 'backhand', 'forehand']
    }

    :param df:
    :param cols_dtype:
    :param with_fake_values:
    :return:
    """
    frames = np.arange(0, n_frames)
    fake_positions = [-1] * n_frames
    fake_positions2 = [0] * n_frames
    fake_bool = [False] * n_frames
    if with_fake_values:
        data = {
            'frame': frames,
            'x': fake_positions,
            'y': fake_positions
        }
        df = pd.DataFrame(data=data)

    for col in ['x', 'y', 'frame']:
        df[col] = df[col].astype('int32')

    for dtype, columns in cols_dtype.items():
        if dtype == 'int':
            fake = np.array([0] * n_frames)
            for col in columns:
                if with_fake_values:
                    df[col] = fake
                df[col] = df[col].astype('int')
        elif dtype == 'bool':
            for col in columns:
                if with_fake_values:
                    df[col] = fake_bool
                df[col] = df[col].astype(bool)
    return df


def save_data(df, save_path):
    df = df.sort_values(by=['frame'])
    df.to_csv(save_path, index=False)
    print(f"data saved automatically in {save_path}")
    return df


if __name__ == '__main__':
    # CHANGE THIS FOR NEW WORK
    # MAKE SURE YOUR CSV FILE IS SAVED EACH TIME YOU ANNOTATE
    VIDEO_FILE = "E:\\TVConal\\TableTennis\\codes\\data\\annotated\\videos\\train\\bb.mp4"
    CSV_SAVE_PATH = 'E:\\TVConal\\TableTennis\\codes\\data\\annotated\\shot_types\\train'

    # CHANGE THIS FOR NEW WORK
    # if left_player == 1 => forehand - if == 2 => backhand
    # if right_player == 1 => forehand | - if == 2 => backhand
    cols_dtype = {
        'int': ['left_player', 'right_player'],
        'bool': ['exclude', 'exclude_end']
    }

    msg_cols = ['left_player', 'right_player', 'exclude', 'exclude_end']

    cap = cv2.VideoCapture(VIDEO_FILE)
    assert cap.isOpened(), "file is not opened!"
    name = Path(VIDEO_FILE).stem

    save_path = join(CSV_SAVE_PATH, name + '.csv')

    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]

    try:
        df = pd.read_csv(save_path)
        df = init(df, cols_dtype)
        print(f"loading from csv file {save_path}")
    except:
        df = init(None, cols_dtype, with_fake_values=True)
        print(f"failed to load {save_path}. initializing ......")

    current = 0
    frame = to_frame(cap, df, current, n_frames)
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
            frame = to_frame(cap, df, current, n_frames, custom_msg=custom_msg)
            break
        elif key == ord('1'):
            # Toggle between values 0, 1, 2
            col = 'left_player'
            output = ''
            val = df.at[current, col]
            if val == 0:
                df.at[current, col] = 1  # Set as forehand
                output = f'{col}: forehand'
            elif val == 1:
                df.at[current, col] = 2  # Set as backhand
                output = f'{col}: backhand'
            else:
                df.at[current, col] = 0
                output = ''

            print(current, f"{output}")
            frame = to_frame(cap, df, current, n_frames, save=True, custom_msg=output)
        elif key == ord('2'):
            # Toggle between values 0, 1, 2
            col = 'right_player'
            output = ''
            val = df.at[current, col]
            if val == 0:
                df.at[current, col] = 1  # Set as forehand
                output = f'{col}: forehand'
            elif val == 1:
                df.at[current, col] = 2  # Set as backhand
                output = f'{col}: backhand'
            else:
                df.at[current, col] = 0
                output = ''

            print(current, f"{output}")
            frame = to_frame(cap, df, current, n_frames, save=True, custom_msg=output)

        elif key == ord('s'):
            df = save_data(df, save_path)
            custom_msg = "Data is saved ..."
            frame = to_frame(cap, df, current, n_frames, custom_msg=custom_msg)
        elif key == ord("f"):
            try:
                check = int(input('Enter your frame:'))
            except:
                print("not a valid number.")
                check = current
            frame = to_frame(cap, df, check, n_frames)
        elif key == ord('x'):
            # Skip to next frame
            check = current + 1
            frame = to_frame(cap, df, check, n_frames)
        elif key == ord('z'):
            # Skip to previous frame
            check = current - 1
            frame = to_frame(cap, df, check, n_frames)
        elif key == ord('3'):
            # CHANGE (toss_end) FOR NEW WORK
            col = 'exclude'
            df.at[current, col] = False if df.at[current, col] else True
            frame = to_frame(cap, df, current, n_frames, save=True)
            print(current, f" {col}: {df.at[current, col]}")

        elif key == ord('4'):
            # CHANGE (toss_end) FOR NEW WORK
            col = 'exclude_end'
            df.at[current, col] = False if df.at[current, col] else True
            frame = to_frame(cap, df, current, n_frames, save=True)
            print(current, f" {col}: {df.at[current, col]}")

        elif key == 2555904:
            # Go to next unlabeled value (Right Arrow =>)
            column = 'left_player'
            next_frame, msg = go_to_next(df, column=column, value=(1, 2), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg=f'next {column}')
        elif key == 2424832:
            column = 'left_player'
            # Go to next unlabeled value (Left Arrow <=)
            next_frame, msg = go_to_previous(df, column=column, value=(1, 2), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg=f'previous {column}')
        elif key == 2490368:
            # Go to next unlabeled value (Up Arrow <=)
            column = 'right_player'
            next_frame, msg = go_to_next(df, column=column, value=(1, 2), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg=f'next {column}')
        elif key == 2621440:
            # Go to previous unlabeled value (Bottom Arrow)
            column = 'right_player'
            next_frame, msg = go_to_previous(df, column=column, value=(1, 2), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg=f'previous {column}')

