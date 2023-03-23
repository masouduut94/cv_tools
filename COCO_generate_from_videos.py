import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from os.path import isdir, isfile, join
from random import randint, choice
import json
from tqdm import tqdm
plt.rcParams['figure.figsize'] = [15, 10]


"""
{
    "categories": [
        ...
        {
            "id": 2,
            "name": "cat",
            "supercategory": "animal"
        },
        ...
    ],
    "images": [
        {
            "id": 1,
            "file_name": "<filename0>.<ext>",
            "height": 480,
            "width": 640,
        },
        ...
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 2,
            "bbox": [260, 177, 231, 199],
            "score": 0.95,
            "area": 45969,
            "iscrowd": 0
        },
        ...
    ]
}

XYXY_ABS = 0
XYWH_ABS = 1

"""

categories = [
        {
            "id": 1,
            "name": "ball",
            "supercategory": "ball"
        },
]

base_path = '/mnt/disk1/tvc_datasets/table_tennis/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/training/'
img_path = join(base_path, 'test/images')

img_id = 1
annot_id = 1

train_annotations = []
train_images = []

test_annotations = []
test_images = []

for i in range(1, 6):
    csv_path = f'annotations/game_{i}/extract_match{i}.csv'
    video_file = join(base_path, f'videos/game_{i}.mp4')
    vdo = cv2.VideoCapture(video_file)
    df = pd.read_csv(csv_path)
    total = len(df)

    pbar = tqdm(total=total)
    train_ratio = int(0.8*total)
    for j, item in df.iterrows():
        temp_img = {}
        temp_annot = {}
        img_is_background = True if item.x == -1 else False
        if img_is_background:
            continue
        # process image
        vdo.set(cv2.CAP_PROP_POS_FRAMES, item.frame_no+1)
        _, frame = vdo.retrieve()
        name = f'match_{i}_frame_{item.frame_no}.png'
        
        # train/test separation
        if j < train_ratio:
            path = join('images/train', name)
            path_to_save = join(base_path, path)
        else:
            path = join('images/test', name)
            path_to_save = join(base_path, path)
        cv2.imwrite(path_to_save, frame)
        # process the image json
        temp_img = {
            'id': img_id,
            "file_name": path,
            "height": frame.shape[0],
            "width": frame.shape[1],
        }
        
        # train/test separation
        if j < train_ratio:
            train_images.append(temp_img)
        else:
            test_images.append(temp_img)
            
        # process the annotations
        x = int(item.x)
        y = int(item.y)
        x1 = x - 15
        y1 = y - 15
        w = 30
        h = 20
        img = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
        temp_annot = {
            "id": annot_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [x1, y1, w, h],
            "bbox_mode": 1,
            "area": w*h,
            "iscrowd": 0
        }

        # increment only when there exist the annotations
        annot_id += 1
        if j < train_ratio:
            train_annotations.append(temp_annot)
        else:
            test_annotations.append(temp_annot)
                
        img_id += 1
        pbar.update(1)
        
        
        
train_js = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}

test_js = {
    'images': test_images,
    'annotations': test_annotations,
    'categories': categories
   
}

json.dump(train_js, open('images/train.json', 'w'))
json.dump(test_js, open('images/test.json', 'w'))

