from typing import Tuple

import cv2
from pathlib import Path
from os.path import join
import numpy as np
import pandas as pd

"""
Data must be like this:

'shot_type': 'push', 'lob', 'loop', 'flick'
'hand_type': 'back-hand' 'fore-hand'
'player': 'left_player' - 'right_player'

frame | x | y | back-hand/fore-hand/serve | push/lob/loop/flick
"""

SKIP1 = 1
SKIP2 = 10
SKIP_WHEEL = 15
SKIP3 = 200
INTEGER_DEFAULT = -1

msg_cols = ['player', 'hand_type', 'shot_type', 'exclude', 'exclude_end']

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


def to_frame(cap, df, current_fno, total_frame, save_path, save=False, custom_msg=None):
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
        message, is_right = init_message(df, current, msg_cols, custom_msg)
        if len(message):
            print("MESSAGE:   ", " | ".join(message))
    else:
        df = save_data(df, save_path)
        print("frame index bigger than number of frames.")
    if not ret:
        return None
    else:
        cv2.putText(frame, f'Frame: {current}/{total_frame}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if len(message):
            if is_right:
                start_width = (int(cap.get(3)) * 3 // 4)
            else:
                start_width = 300
            colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
            for i, (item, color) in enumerate(zip(message, colors)):
                cv2.putText(frame, item, (start_width, 300 + (i * 50)), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
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


def init_message(df, index, columns, custom_msg=None):
    # data = df.iloc[index]
    items = []
    is_right_player = None
    if df.at[index, 'player'] == 0:
        is_right_player = False
    elif df.at[index, 'player'] == 1:
        is_right_player = True

    for col in columns:
        if str(df.dtypes[col]).startswith('int'):
            val = df.at[index, col]
            if val == -1:
                continue
            reversed_dict = {val: key for (key, val) in all_dicts[col].items()}
            items.append(f"{col}: {reversed_dict[val]}")
            # st += f"{col}: {reversed_dict[val]} | "

        elif str(df.dtypes[col]).startswith('bool'):
            val = df.at[index, col]
            if val:
                items.append(f'{col} flag')
                # st += f'{col} flag |'

    if custom_msg:
        items.insert(0, custom_msg)

    # st = custom_msg if custom_msg is not None else st
    # print(st)

    return items, is_right_player


def init(df: pd.DataFrame, cols_dtype: dict, n_frames, with_fake_values: bool = False):
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
            for col in columns:
                if with_fake_values:
                    df[col] = fake_positions
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


def toggle(df: pd.DataFrame, current: int, col: str, col_dict: dict):
    current_val = df.at[current, col]
    values = list(col_dict.values())
    keys = list(col_dict.keys())
    reversed_dict = {k: v for k, v in zip(values, keys)}
    next_val = current_val + 1
    try:
        reversed_dict[next_val]
    except KeyError:
        next_val = INTEGER_DEFAULT
    df.at[current, col] = next_val
    return reversed_dict[next_val]


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
    VIDEO_FILE = "E:\\TVConal\\TableTennis\\codes\\data\\annotated\\videos\\train\\dd.mp4"
    CSV_SAVE_PATH = 'E:\\TVConal\\TableTennis\\codes\\data\\annotated\\shot_types\\train'

    cap = cv2.VideoCapture(VIDEO_FILE)
    assert cap.isOpened(), "file is not valid!"
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
            frame = to_frame(cap, df, next_frame, n_frames, save_path=CSV_SAVE_PATH, custom_msg=f'next {column}')
        print(f"loading from csv file {save_path}")
    except FileNotFoundError:
        df = init(None, cols_dtype, n_frames=n_frames, with_fake_values=True)
        print(f"failed to load {save_path}. initializing ......")

    frame = to_frame(cap, df, current, n_frames, save_path=save_path)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        # frame = cv2.resize(frame, (w, h))
        # frame = cv2.resize(frame, (w//2, h//2))
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
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,
                                 custom_msg=f'jumping next {column}')
        elif key == 2424832:
            column = 'player'
            # Go to next unlabeled value (Left Arrow <=)
            next_frame, msg = go_to_previous(df, column=column, value=(0, 1), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,
                                 custom_msg=f'jumping previous {column}')
        elif key == 2490368:
            # Go to next unlabeled value (Up Arrow <=)
            column = 'shot_type'
            next_frame, msg = go_to_next(df, column=column, value=tuple(range(0, 6)), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,
                                 custom_msg=f'jumping next {column}')
        elif key == 2621440:
            # Go to previous unlabeled value (Bottom Arrow)
            column = 'shot_type'
            next_frame, msg = go_to_previous(df, column=column, value=tuple(range(0, 5)), current=current)
            if next_frame is not None:
                frame = to_frame(cap, df, next_frame, n_frames, save_path=save_path,
                                 custom_msg=f'jumping previous {column}')
