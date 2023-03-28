from tqdm import tqdm
from pytube import YouTube

links = [
    'https://www.youtube.com/watch?v=AAN1vhtvMpk',
    'https://www.youtube.com/watch?v=tpWAQtxOZEk',
    'https://www.youtube.com/watch?v=yyixYTLP1Z4',
    'https://www.youtube.com/watch?v=Y5auDSmyNtg',
    'https://www.youtube.com/watch?v=c7m2DWJqmhY',
    'https://www.youtube.com/watch?v=GzMXrFt0mvM',
    'https://www.youtube.com/watch?v=8G3jFJ6AQ4Y'
]

volley_ball_rear_camera_views = [
    'https://www.youtube.com/watch?v=GmsP3ErGiLk',
    'https://www.youtube.com/watch?v=HSMMPl3iHiE',
    'https://www.youtube.com/watch?v=P9bmiyNQoMs',
    'https://www.youtube.com/watch?v=ESDQaH2wKL0',
    'https://www.youtube.com/watch?v=IYwAGz7BhXY',
    'https://www.youtube.com/watch?v=DWq7lRpvaLw',
    'https://www.youtube.com/watch?v=hHXyJ-Qm-XE'

]

for l in tqdm(links):
    yt = YouTube(l)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    # stream = yt.streams.first()
    stream.download('./')

