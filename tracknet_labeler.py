import cv2
from pathlib import Path
from os.path import join
import numpy as np
import pandas as pd

"""
Data must be like this:
frame | x | y | toss | end_toss | exclude | exclude-end
"""

def check_fno(fno, total_frame):
    """
    check if suggested frame number is not invalid based on video number of frames.
    Args:
        fno:
        total_frame:

    Returns:

    """
    if fno <= 0:
        print('\nInvaild !!! Jump to first image...')
        return False
    elif fno > total_frame:
        print(f"\n maximum frames = {total_frame}")
    else:
        print(f"Frame set to: {fno}")
        return True


def to_frame(cap, df, n, total_frame):
    print('current frame: ', n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()

    message = init_message(df, n, msg_columns)

    if not ret:
        return None
    else:
        cv2.putText(frame, f'Frame: {n}/{total_frame}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if message != '':
            cv2.putText(frame, message, (100, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        item = df.iloc[current]
        if item.x != -1:
            color = (0, 0, 255)
            x = item.x
            y = item.y
            cv2.circle(frame, (x, y), 5, color, thickness=-1)
        return frame


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global data, cap, current
    global frame

    if event == cv2.EVENT_LBUTTONDOWN:
        df.at[current, 'x'] = x
        df.at[current, 'y'] = y
        frame = to_frame(cap, df, current, n_frames)


def init_message(df, index, columns):
    # data = df.iloc[index]
    st = ''

    for col in columns:
        if df.at[index, col]:
            st += f'| {col}'

    return st


def init(df, bool_cols=(), int_cols=(), str_cols=()):
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    for col in str_cols:
        df[col] = df[col].astype('str')

    for col in int_cols:
        df[col] = df[col].astype('int')

    # df['frame'] = df['frame'].astype('int')
    # df['x'] = df['x'].astype('int')
    # df['y'] = df['y'].astype('int')
    # df['shot_type'] = df['shot_type'].astype('int')
    # df['bounce_point'] = df['bounce_point'].astype('bool')
    # df['net'] = df['net'].astype('bool')
    # df['serve_toss'] = df['serve_toss'].astype('bool')
    # df['rally_start'] = df['rally_start'].astype('bool')
    # df['rally_end'] = df['rally_end'].astype('bool')
    return df


if __name__ == '__main__':
    videofile = "E:/TVConal/TableTennis/codes/videos/behind_cam.mp4"

    columns = [
        'x',
        'y',
        'frame',
        'toss',
        'toss_end',
        'exclude',
        'exclude_end'
    ]
    # columns = [
    #     "frame", "toss", "rally_start",
    #     "rally_end", "exclude", "exclude_end"
    # ]
    msg_columns = [
        "toss", 'toss_end', "exclude", "exclude_end"
    ]

    bool_cols = (
        "toss", 'toss_end', "exclude", "exclude_end"
    )
    int_cols = ('x', 'y', 'frame',)

    #
    # cols = [
    #     'frame', 'x', 'y', 'shot_type',
    #     'serve_toss', 'rally_start', 'rally_end',
    #     'bounce_point', 'net'
    # ]
    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), "file is not opened!"
    name = Path(videofile).stem
    filepath = 'E:/TVConal/TableTennis/codes/labels'
    save_path = join(filepath, name + '.csv')

    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]

    try:
        df = pd.read_csv(save_path)
        df = init(df, bool_cols=bool_cols, int_cols=int_cols)
    except:
        frames = np.arange(0, n_frames)
        shot_types = [-1] * n_frames
        xy = [-1] * n_frames
        z = [False] * n_frames
        # data = {
        #     'frame': frames,
        #     'x': xy,
        #     'y': xy,
        #     'shot_type': xy,
        #     'serve_toss': z,
        #     'rally_start': z,
        #     'rally_end': z,
        #     'bounce_point': z,
        #     'net': z
        # }

        data = {
            'frame': frames,
            'x': xy,
            'y': xy,
            'toss': z,
            'toss_end': z,
            'exclude': z,
            'exclude_end': z
        }

        df = pd.DataFrame(data=data)
        df = init(df, bool_cols=bool_cols, int_cols=int_cols)

    current = 0
    frame = to_frame(cap, df, current, n_frames)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    # cv2.setMouseCallback("image", click_and_crop)

    SKIP1 = 1
    SKIP2 = 10
    SKIP3 = 500

    while True:
        # frame = cv2.resize(frame, (w, h))
        cv2.imshow("image", frame)
        key = cv2.waitKeyEx(1)  # & 0xFF
        # print(key)
        if key == 27:  # Esc
            df = df.sort_values(by=['frame'])
            df.to_csv(save_path, index=False)
            break
        elif key == ord('1'):
            df.at[current, 'toss'] = False if df.at[current, "toss"] else True
        elif key == ord('2'):
            df.at[current, 'toss_end'] = False if df.at[current, "toss_end"] else True
        elif key == ord('3'):
            df.at[current, 'exclude'] = False if df.at[current, "exclude"] else True
        elif key == ord('4'):
            df.at[current, 'exclude_end'] = False if df.at[current, "exclude_end"] else True
        # elif key == ord('5'):
        #     df.at[current, 'serve_toss'] = False if df.iloc[current].serve_toss else True
        # elif key == ord('6'):
        #     if df.iloc[current].shot_type == -1:
        #         df.at[current, 'shot_type'] = 0
        #         print("shot type: ForeHand")
        #     elif df.iloc[current].shot_type == 0:
        #         df.at[current, 'shot_type'] = 1
        #         print("shot type: Back-Hand")
        #     else:
        #         df.at[current, 'shot_type'] = -1
        #         print("shot type: Disabled")

        # elif key == ord('l'):
        #     # Restart the values
        #     df.at[current, 'x'] = -1
        #     df.at[current, 'y'] = -1
        elif key == ord('s'):
            df = df.sort_values(by=['frame'])
            df.to_csv(save_path, index=False)
            print(f"saved csv in {save_path}")
        elif key == ord("d"):  # jump 1 frame
            check = current + SKIP1
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
        elif key == ord("e"):  # => jump next 20 frame
            check = current + SKIP2
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
        elif key == ord("c"):  # <= jump next 300 frames
            check = current + SKIP3
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
        elif key == ord("a"):  # jump back 1 frame
            check = current - SKIP1
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
        elif key == ord("q"):  # jump back 20 frame
            check = current - SKIP2
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
        elif key == ord("z"):  # jump back 300 frame
            check = current - SKIP3
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
        elif key == ord("f"):  # jump back 300 frame
            try:
                check = int(input('Enter your frame:'))
            except:
                print("not a valid number.")
                check = current
            is_ok = check_fno(check, n_frames)
            if is_ok:
                current = check
            frame = to_frame(cap, df, current, n_frames)
