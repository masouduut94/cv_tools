from typing import Union

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


def go_to_next(df: pd.DataFrame, column: str, value: Union[bool, int], current: int, return_last=False):
    next_indexes = df.index[df.frame > current]
    msg = f"no more `{column}` after current value"
    if not len(next_indexes):
        print(msg)
        return None, msg

    temp = df[(df[column] == value) & (df.index.isin(next_indexes))]
    if len(temp):
        if return_last:
            return temp.iloc[-1].frame, ""
        return temp.iloc[0].frame, ""
    else:
        print(msg)
        return None, msg


def go_to_previous(df: pd.DataFrame, column: str, value: Union[bool, int], current: int, return_first=False):
    next_indexes = df.index[df.frame < current]
    msg = f"no more `{column}` before current value"
    if not len(next_indexes):
        print(msg)
        return None, msg

    temp = df[(df[column] == value) & (df.index.isin(next_indexes))]
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
        if df.at[index, col]:
            st += f'| {col}'

    st = custom_msg if custom_msg is not None else st

    return st


def init(df: pd.DataFrame, cols_dtype: dict, with_fake_values: bool = False):
    """
    cols_dtype = {
        'bool': ['toss', 'end_toss', 'exclude', 'end_exclude']
    }

    :param df:
    :param cols_dtype:
    :param with_fake_values:
    :return:
    """
    frames = np.arange(0, n_frames)
    fake_positions = [-1] * n_frames
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
        if dtype == 'int32':
            fake = np.array([-1] * n_frames)
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
    VIDEO_FILE = "E:/TVConal/TableTennis/codes/data/to_annotate/bb.mp4"
    CSV_SAVE_PATH = 'E:/TVConal/TableTennis/codes/data/to_annotate/'

    # CHANGE THIS FOR NEW WORK
    cols_dtype = {
        'bool': ['toss', 'toss_end', 'exclude', 'exclude_end']
    }

    msg_cols = ['toss', 'toss_end', 'exclude', 'exclude_end']

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
            # CHANGE (toss) FOR NEW WORK
            df.at[current, 'toss'] = False if df.at[current, "toss"] else True
            print(current, f" toss: {df.at[current, 'toss']}")
            frame = to_frame(cap, df, current, n_frames, save=True)
        elif key == ord('2'):
            # CHANGE (toss_end) FOR NEW WORK
            df.at[current, 'toss_end'] = False if df.at[current, "toss_end"] else True
            frame = to_frame(cap, df, current, n_frames, save=True)
            print(current, f" toss_end: {df.at[current, 'toss_end']}")
        elif key == ord('3'):
            # CHANGE (exclude) FOR NEW WORK
            df.at[current, 'exclude'] = False if df.at[current, "exclude"] else True
            frame = to_frame(cap, df, current, n_frames, save=True)
            print(current, f" exclude: {df.at[current, 'exclude']}")
        elif key == ord('4'):
            # CHANGE (exclude_end) FOR NEW WORK
            df.at[current, 'exclude_end'] = False if df.at[current, "exclude_end"] else True
            frame = to_frame(cap, df, current, n_frames, save=True)
            print(current, f" exclude_end: {df.at[current, 'exclude_end']}")
        elif key == ord('s'):
            df = save_data(df, save_path)
            custom_msg = "Data is saved ..."
            frame = to_frame(cap, df, current, n_frames, custom_msg=custom_msg)
        elif key == ord('t'):
            # Skip to next `toss` (if annotated before)
            next_frame, msg = go_to_next(df, column='toss', value=True, current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg='jumping to next toss')
            else:
                frame = to_frame(cap, df, current, n_frames, custom_msg=msg)
        elif key == ord('g'):
            # Skip to previous `toss` (if annotated before)
            prev_frame, msg = go_to_previous(df, column='toss', value=True, current=current)
            if prev_frame is not None:
                frame = to_frame(cap, df, prev_frame, n_frames, custom_msg='jumping to previous toss')
            else:
                frame = to_frame(cap, df, current, n_frames, custom_msg=msg)

        elif key == ord('q'):
            # Skip to next `toss_end` (if annotated before)
            next_frame, msg = go_to_next(df, column='toss_end', value=True, current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg='jumping to next toss_end')
            else:
                frame = to_frame(cap, df, current, n_frames, custom_msg=msg)
        elif key == ord('a'):
            # Skip to previous `toss_end` (if annotated before)
            prev_frame, msg = go_to_previous(df, column='toss_end', value=True, current=current)
            if prev_frame is not None:
                frame = to_frame(cap, df, prev_frame, n_frames, custom_msg='jumping to previous toss_end')
            else:
                frame = to_frame(cap, df, current, n_frames, custom_msg=msg)
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

        elif key == 2555904:
            # Go to next unlabeled value (Page Down or | arrow)
            next_frame, msg = go_to_next(df, column='toss', value=True, current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg='jumping to next toss')
        elif key == 2424832:
            # Go to next unlabeled value (Page Up | <- arrow)
            next_frame, msg = go_to_previous(df, column='toss', value=True, current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, custom_msg='jumping to next toss')