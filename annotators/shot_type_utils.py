import cv2
from os.path import join
from typing import Tuple, List
import pandas as pd
import numpy as np

SKIP1 = 1
SKIP2 = 10
SKIP_WHEEL = 15
SKIP3 = 200
INTEGER_DEFAULT = -1

msg_cols = ['player', 'hand_type', 'shot_type', 'exclude', 'exclude_end']


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
        message = init_message(df, current, msg_cols, custom_msg)
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
            colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
            for i, (item, color) in enumerate(zip(message, colors)):
                cv2.putText(frame, item, (100, 300+(i*50)), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
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
    st = ''

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

    return items


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
