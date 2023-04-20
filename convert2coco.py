import cv2
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os.path import isfile, join, isdir
from os import makedirs
import matplotlib.pyplot as plt
import shutil
import random
from apply_unique_names import preprocess_images

random.seed(1394)
plt.rcParams['figure.figsize'] = [20, 18]

STD_CLASSES = {
    1: 'ball',
}

def read_classes(path):
    cls_json = json.load(open(path))
    classes = {}
    for item in cls_json:
        classes[item['id']] = item['name'].lower()
    return classes


BASE_DIR = 'DATA'
all_subdirs = [item.as_posix() for item in list(Path(BASE_DIR).glob('*')) if isdir(item.as_posix())]

APPLY_UNIQUIFY = False
CLASSES_UNIQUIFY = False

# Some of files have the same name between all directories.
# Maybe we could also remove the shot_types with the background.
if APPLY_UNIQUIFY:
    for path in tqdm(all_subdirs):
        preprocess_images(path)

    print('Images uniquified!')

# This is due to some classes are not mapped to their standard ids:
# For example we expect {1: 'batsman'} but the shot_types are {2: 'batsman'} when labeling the items.

if CLASSES_UNIQUIFY:
    for base_path in tqdm(all_subdirs):
        p = join(base_path, 'classes.json')
        classes_loaded = read_classes(p)
        annotations = join(base_path, 'annotations.json')
        annots = json.load(open(annotations))
        for k, annot in annots.items():
            instances = []
            for instance in annot['instances']:
                old_class_id = instance['classId']
                mapping = {}
                for i, v in classes_loaded.items():
                    for ii, vv in STD_CLASSES.items():
                        if v == vv:
                            mapping[i] = ii
                instance['classId'] = mapping[old_class_id]
                instances.append(instance)
            annot['instances'] = instances
        with open(annotations, 'w') as file_:
            json.dump(annots, file_, indent=2, sort_keys=True)

        with open(p, 'w') as file_:
            classes_js = [
                    {
                        "attribute_groups": [
                            {
                                "id": 1,
                                "name": "confidence",
                                "is_multiselect": True,
                                "attributes": [
                                    {
                                        "id": 1,
                                        "name": "not_sure"
                                    }
                                ],
                                "opened": True,
                                "hasChanges": True
                            }
                        ],
                        "color": "#32a852",
                        "id": 1,
                        "name": "batsman",
                        "opened": True
                    },
                    {
                        "attribute_groups": [
                            {
                                "id": 2,
                                "name": "confidence",
                                "is_multiselect": False,
                                "attributes": [
                                    {
                                        "id": 2,
                                        "name": "not_sure"
                                    }
                                ],
                                "opened": True,
                                "hasChanges": True
                            }
                        ],
                        "color": "#26bf9e",
                        "id": 2,
                        "name": "bat",
                        "opened": True
                    },
                    {
                        "attribute_groups": [
                            {
                                "id": 3,
                                "name": "confidence",
                                "is_multiselect": False,
                                "attributes": [
                                    {
                                        "id": 3,
                                        "name": "not_sure"
                                    }
                                ],
                                "opened": True,
                                "hasChanges": True
                            }
                        ],
                        "color": "#694882",
                        "id": 3,
                        "name": "helmet",
                        "opened": True
                    }
                ]

            json.dump(classes_js, file_)


all_images = list(Path(BASE_DIR).rglob("*.png"))
all_annotations = list(Path(BASE_DIR).rglob('*tations.json'))
all_classes = list(Path(BASE_DIR).rglob('classes.json'))

#
#
# # Some classes dont abide by the standard classes
# for classes_config in all_classes:
#
#     base_dir = classes_config.parent
#     ####### start process of mapping classes to standard classes



classes = {
    1: 'batsman',
    2: 'bat',
    3: 'helmet'
}

# Check if we got images with the same name
# not_repeating_names = [item.name for item in all_images]
# images_with_same_name = []
#
# for item in not_repeating_names:
#     if not_repeating_names.count(item) == 2:
#         images_with_same_name.append(item)
#
# print(images_with_same_name)
# # Make sure all directories have the same class.json
# all_class_id_and_name = []
# include_bg = [] # the ones include background
# for item in all_classes:
#     temp_json = json.load(open(item.as_posix()))
#     temp = {}
#     for ii in temp_json:
#         temp[ii['id']] = ii['name'].lower()
#     all_class_id_and_name.append(temp)
#     if len(list(temp.keys())) > 3:
#         include_bg.append(item.as_posix())
#
# print("the ones that include background in classes json")
# for item in include_bg:
#     print(item)
#
# assert all(x == all_class_id_and_name[0] for x in all_class_id_and_name)


# Union the annotations with full address to images as keys of dictionary
annots_union = {}

for annot in all_annotations:
    js = json.load(open(annot.as_posix()))
    base_dir = annot.parent
    try:
        del js['___sa_version___']
    except KeyError:
        pass
    keys_to_remove = []
    js_with_full_path_keys = {}
    for key, value in js.items():
        full_address = join(base_dir, 'images', key)
        js_with_full_path_keys[full_address] = value
        keys_to_remove.append(key)

    annots_union.update(js_with_full_path_keys)

# if background is exluded from shot_types then
# delete the images with background annotation
#
# images_to_remove = []
# for key, value in annots_union:
#     if value['instances']['classId'] == 4:
#         images_to_remove.append(key)


# Check if all images are annotated.
if len(list(annots_union)) != len(all_images):
    all_images_keys = [item.name for item in all_images]
    all_annots_keys = list(annots_union.keys())
    for item in all_annots_keys:
        if item not in all_images_keys:
            print(item)
else:
    print("all images has corresponding annotations! OK")

"""
These two modules are utility modules for better handling the images and its annotations
to darknet and coco formats.


Read the detectron2 docs here:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode


XYXY_ABS= 0 [x]
XYWH_ABS= 1 [x]
XYXY_REL= 2
XYWH_REL= 3
XYWHA_ABS= 4


"""
BBOX_MODE = {
    'XYXY_ABS': 0,
    'XYWH_ABS': 1
}


class Bbox:
    """
    This class is implemented to give better control over annotations bounding boxes
    and its corresponding statistics. It can output bounding box statistics to
    darknet format and coco format.
    """

    def __init__(self, x1, y1, x2, y2, category_id=1, id_=None, image_id=None):
        self.id = id_
        self.x1 = int(min([x1, x2]))
        self.y1 = int(min([y1, y2]))
        self.x2 = int(max([x1, x2]))
        self.y2 = int(max([y1, y2]))
        self.image_id = image_id
        self.category_id = int(category_id)

    def __repr__(self):
        return f"""(Bbox(id={self.id}, x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, width={abs(self.x1 - self.x2)}, height={abs(self.y1 - self.y2)}, category_id={self.category_id})"""

    def to_coco(self, bbox_mode=BBOX_MODE['XYXY_ABS']):
        """
        Convert information to coco format
        Args:
            bbox_mode: to output bbox as xyxy or xywh format.

        Returns(dict):

        """
        width = abs(self.x1 - self.x2)
        height = abs(self.y1 - self.y2)
        if bbox_mode == BBOX_MODE['XYWH_ABS']:
            return {
                'id': self.id,
                'category_id': self.category_id,
                'image_id': self.image_id,
                'iscrowd': 0,
                'bbox': [self.x1, self.y1, width, height],
                "area": width * height,
                "bbox_mode": BBOX_MODE['XYWH_ABS'],
            }

        else:
            return {
                'id': self.id,
                'category_id': self.category_id,
                'image_id': self.image_id,
                'iscrowd': 0,
                'bbox': [self.x1, self.y1, self.x2, self.y2],
                "area": width * height,
                "bbox_mode": BBOX_MODE['XYXY_ABS'],
            }

    def to_darknet(self):
        """
        prepares the x_center, y_center, width and height of
        the bounding box.
        Notes:
            you should be aware that we have to divide these
            values to the image_width and image_height to get
            a float number between 0 and 1. we do that on
            ImageAnnotation module.
        Returns:

        """
        width = abs(self.x1 - self.x2)
        height = abs(self.y1 - self.y2)

        x_cen = (self.x1 + width / 2)
        y_cen = (self.y1 + height / 2)

        return {
            'category_id': self.category_id,
            'x_cen': x_cen,
            'y_cen': y_cen,
            'width': width,
            'height': height
        }


class ImageAnnotations(object):
    """
    This module provides utility functions over image and its annotations.
    It can output json to COCO format and darknet.
    Capabilities:
        - It can filter out the categories of objects in the output json.
        - It can help you plot and test the objects in the image to check
             out the annotations. you can also filter certain objects in
             plot if you want
    """
    CATEGORIES = list(classes.keys())

    def __init__(self, image_id, file_name):
        self.image_id = image_id
        self.file_name = file_name
        assert isfile(file_name), f"file {file_name} does not exist"
        image = cv2.imread(file_name)
        self.height, self.width = image.shape[:2]
        self.annotations = []
        del image

    def add_annotation(self, bbox: Bbox):
        self.annotations.append(bbox)

    def __len__(self):
        return len(self.annotations)

    def __repr__(self):
        return f"""(ImageAnnotations object: file_name={self.file_name}, image_id={self.image_id}, width={self.width}, height={self.height}, n_annotations={len(self)})"""

    def to_coco_format(self, allowed_categories=()) -> dict:
        """
        Prepares the image details and the annotations on an image.
        Notes:
            The annotations must be assigned a unique integer. We have to
            do that only when we have access to all the annotations of all
            images. So there is a need for loop over all images and their
            annotations to assign unique integer for both image and
            corresponding annotations.
        Returns:

        """
        if not bool(allowed_categories):
            allowed_categories = self.CATEGORIES

        annots_details = []
        for annot in self.annotations:
            coco_info = annot.to_coco(BBOX_MODE['XYWH_ABS'])
            cat_id = coco_info['category_id']
            if cat_id not in allowed_categories:
                continue
            coco_info['image_id'] = self.image_id
            annots_details.append(coco_info)
        image_info = {
            'id': self.image_id,
            'width': self.width,
            'height': self.height,
            'file_name': self.file_name
        }
        return {
            'image': image_info,
            'annotations': annots_details
        }

    def to_darknet_format(self, base_dir=None, allowed_categories=()):
        txt = ''
        if not bool(allowed_categories):
            allowed_categories = self.CATEGORIES

        """
        Darknet shot_types starts from 0 to n-1 classes
        So if we assume allowed_categories are [1, 4, 6] then all categories are sth like 
        [1, 2, ..., 10] for example. then we need to map [1, 4, 6] to [0, 1, 2] which 
        darknet is supposed to work with.

        """
        categories = []
        for k, v in classes.items():
            if k in allowed_categories:
                categories.append(k)

        map_cats_to_darknet_format = {item: i for i, item in enumerate(categories)}  # {1: 0, 4: 1, 6: 2}

        for annot in self.annotations:
            info = annot.to_darknet()
            category_id = info['category_id']
            if category_id not in allowed_categories:
                continue
            cat_id = map_cats_to_darknet_format[category_id]
            x_cen = info['x_cen'] / self.width
            y_cen = info['y_cen'] / self.height
            width = info['width'] / self.width
            height = info['height'] / self.height

            txt += f'{cat_id} {x_cen:.2f} {y_cen:.2f} {width:.2f} {height:.2f}\n'
        txt = txt[:-2]  # omitting last \n
        if base_dir:
            filename = Path(self.file_name).stem
            with open(join(base_dir, filename + '.txt'), 'w') as file:
                file.write(txt)
            return

        return {
            'file_name': self.file_name,
            'label_string': txt,
            'labels_included': {k: v for k, v in classes.items() if k in allowed_categories},
            'is_background': 1 if len(txt) == 0 else 0
            # when some image does not include any image, its totally background
        }

    # TODO: Define constant categories
    def coco_plot(self, allowed_categories=()):
        image = cv2.imread(self.file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = np.random.randint(low=0, high=255, size=(3, len(classes))).tolist()
        colormap = {item: colors[i] for i, item in enumerate(list(classes.values()))}
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 3
        if not bool(allowed_categories):
            allowed_categories = self.CATEGORIES

        for annot in self.annotations:
            annot_info = annot.to_coco(bbox_mode=BBOX_MODE['XYXY_ABS'])
            category_id = annot_info['category_id']
            if category_id not in allowed_categories:
                continue
            x1, y1, x2, y2 = annot_info['bbox']
            category_name = classes[
                category_id]  # Mapping of category_id in superannotate starts from 1 but we insert categories from 0 because of coco standard format
            image = cv2.rectangle(image, (x1, y1), (x2, y2),
                                  color=colormap[category_name], thickness=3)
            text_org = (x1 - 5, y1 - 5)
            image = cv2.putText(image, f"{category_name.upper()}",
                                org=text_org, fontFace=font_face,
                                fontScale=font_scale, color=colormap[category_name],
                                thickness=thickness)
        return image

    @staticmethod
    def get_coco_categories(allowed_categories=()):
        categories = []
        temp = {}
        if not bool(allowed_categories):
            allowed_categories = list(classes.keys())
        
        for k, label in classes.items():
            if k in allowed_categories:
                temp['supercategory'] = label
                temp['id'] = k
                temp['name'] = label
                categories.append(temp.copy())
        return categories


"""
We register all images and bounding boxes with these stuff:
image:
    id
    width
    height
    filename (The address where it is going to be uploaded)

annotations:
    id
    image_id
    category_id
    iscrowd=0
    bbox ([x1, y1, x2, y2] or [x1, y1, width, height])
"""
bbox_index = 0
images_list = []
CLASSES_TO_INCLUDE = [1, 2, 3]  # The class ids that we are interested to involve in the object detection task.

for image_id, (file_name, annotations_info) in enumerate(tqdm(annots_union.items())):
    # We need two arguments for initializing the ImageAnnotations
    # image_id and filename
    image_and_annotation = ImageAnnotations(image_id=image_id, file_name=file_name)

    for instance in annotations_info['instances']:
        if (instance['type'] == 'bbox') and (instance['classId'] in CLASSES_TO_INCLUDE):
            points = instance['points']
            category_id = instance['classId']  # Classes has to start from 1
            x1, x2, y1, y2 = points['x1'], points['x2'], points['y1'], points['y2']
            bbox = Bbox(x1, y1, x2, y2, category_id, id_=bbox_index,
                        image_id=image_id)
            bbox_index += 1
            image_and_annotation.add_annotation(bbox)
    images_list.append(image_and_annotation)

categories_to_plot = [1, 2]  # batsman and bat

rnd = random.choice(images_list)
plt.imshow(rnd.coco_plot(allowed_categories=categories_to_plot))
plt.show()
print("testing plot on image: OK")

allowed_categories = (1, 2)
js = rnd.to_coco_format(allowed_categories=allowed_categories)
print(js)
print("testing categories fileteration: OK")

js = rnd.to_darknet_format(allowed_categories=allowed_categories)
text = js['label_string'].split('\n')

if js['is_background'] == 0:
    for item in text:
        print(item)
print("testing categories fileteration on darknet output: OK")


def coco_prepare(new_path_base_dir: str, image_annot_list: list, n_data: int,
                 json_filename: str, all_categories: list, allowed_categories=(1, 2, 3)):
    """
    This function is designed to create directories for train and validation in coco format.
    Args:
        new_path_base_dir: I use train/validation to pass this arg
        image_annot_list:
        n_data:
        json_filename:
        all_categories:
        allowed_categories:

    Returns:

    """
    coco_images = []
    coco_annotations = []

    if not isdir(new_path_base_dir):
        makedirs(new_path_base_dir)
    i = 0
    while i < n_data:
        image_annot = image_annot_list.pop()
        output_js = image_annot.to_coco_format(allowed_categories)
        image_info = output_js['image']
        # Copying images in new path (train/val/test)
        old_path = Path(image_info['file_name'])
        new_path = join(new_path_base_dir, old_path.name)
        image_info['file_name'] = new_path
        shutil.copy2(old_path.as_posix(), new_path)
        annotation_info = output_js['annotations']
        for item in annotation_info:
            coco_annotations.append(item)
        coco_images.append(image_info)
        i += 1

    coco_annotations = sorted(coco_annotations, key=lambda x: x['id'])

    js = {
        'images': coco_images,
        'categories': all_categories,
        'annotations': coco_annotations
    }

    with open(json_filename, 'w') as file:
        json.dump(js, file, indent=2, sort_keys=True)


def darknet_prepare(new_path_base_dir: str, image_annot_list: list, n_data: int, is_train=True, allowed_categories=()):
    """
    Move the images to new path and creates the shot_types for each image in darknet format.

    Args:
        new_path_base_dir:
        image_annot_list:
        n_data:
        is_train:
        allowed_categories:

    Returns:

    """
    base_images_dir = join(new_path_base_dir, 'images')
    base_labels_dir = join(new_path_base_dir, 'shot_types')

    if is_train:
        path_to_cp_images = join(base_images_dir, 'train')
        path_to_cp_labels = join(base_labels_dir, 'train')
    else:
        path_to_cp_images = join(base_images_dir, 'val')
        path_to_cp_labels = join(base_labels_dir, 'val')

    if not isdir(path_to_cp_labels):
        makedirs(path_to_cp_labels)

    if not isdir(path_to_cp_images):
        makedirs(path_to_cp_images)

    i = 0
    while i < n_data:
        image_annot = image_annot_list.pop()
        old_path = Path(image_annot.file_name)
        new_img_path = join(path_to_cp_images, old_path.name)
        shutil.copy2(old_path.as_posix(), new_img_path)
        image_annot.to_darknet_format(base_dir=path_to_cp_labels, allowed_categories=allowed_categories)
        i += 1


######################################## Split to Train/Test ##################################################
to_darknet_flag = False
to_coco_flag = True

random.shuffle(images_list)

SIZE = len(images_list)

train_ratio = 0.8
# val_ratio = 0.2
# test_ratio = 0.1

TRAIN_PATH = 'dataset/train'
VAL_PATH = 'dataset/val'
DARKNET_PATH = 'darknet_dataset'

all_categories = images_list[0].get_coco_categories(allowed_categories=(1,))
train_size = int(train_ratio * SIZE)
test_size = len(images_list) - train_size

if to_coco_flag:
    coco_prepare(TRAIN_PATH, images_list, train_size, 'train.json', all_categories, allowed_categories=(1,))
    coco_prepare(VAL_PATH, images_list, test_size, 'test.json', all_categories, allowed_categories=(1, ))

if to_darknet_flag:
    darknet_prepare(DARKNET_PATH, images_list, train_size, is_train=True, allowed_categories=(1, 2, 3))
    darknet_prepare(DARKNET_PATH, images_list, test_size, is_train=False, allowed_categories=(1, 2, 3))
