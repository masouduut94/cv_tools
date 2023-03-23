"""
These modules come in handy to convert different formats to COCO and Darknet format.

"""
import cv2
import json
import shutil
from tqdm import tqdm
from os import makedirs
from pathlib import Path
import matplotlib.pyplot as plt
from os.path import isfile, join, isdir

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
    CATEGORIES = ['ball']

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

    def to_coco_format(self) -> dict:
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

        annots_details = []
        for annot in self.annotations:
            coco_info = annot.to_coco(BBOX_MODE['XYWH_ABS'])
            cat_id = 1

            coco_info['category_id'] = cat_id
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

    # TODO: Define constant categories
    def coco_plot(self):
        image = cv2.imread(self.file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colormap = (255, 200, 0)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 3

        for annot in self.annotations:
            annot_info = annot.to_coco(bbox_mode=BBOX_MODE['XYXY_ABS'])
            category_id = annot_info['category_id']

            x1, y1, x2, y2 = annot_info['bbox']
            category_name = 'ball'
            image = cv2.rectangle(image, (x1, y1), (x2, y2),
                                  color=colormap, thickness=3)
            text_org = (x1 - 5, y1 - 5)
            image = cv2.putText(image, f"{category_name.upper()}",
                                org=text_org, fontFace=font_face,
                                fontScale=font_scale, color=colormap,
                                thickness=thickness)
        return image

    @staticmethod
    def get_coco_categories(label):
        temp = {
            'supercategory': label,
            'id': 1,
            'name': label
        }
        categories = [temp.copy()]
        return categories


def coco_prepare(new_path_base_dir: str, image_annot_list: list, json_filename: str, label):
    """
    This function is designed to create directories for train and validation in coco format.
    Args:
        new_path_base_dir: I use train/validation to pass this arg
        image_annot_list:
        json_filename:

    Returns:

    """
    coco_images = []
    coco_annotations = []

    if not isdir(new_path_base_dir):
        makedirs(new_path_base_dir)

    print('Copying the images to new directory')
    progressbar = tqdm(total=len(image_annot_list))
    all_categories = image_annot_list[0].get_coco_categories(label)

    while len(image_annot_list):
        image_annot = image_annot_list.pop()
        output_js = image_annot.to_coco_format()
        image_info = output_js['image']
        # Copying images in new path (train/val/test)
        old_path = Path(image_info['file_name'])
        new_path = join(new_path_base_dir, old_path.name)
        image_info['file_name'] = new_path
        shutil.move(old_path.as_posix(), new_path)
        annotation_info = output_js['annotations']
        for item in annotation_info:
            coco_annotations.append(item)
        coco_images.append(image_info)
        progressbar.update(1)

    progressbar.close()
    coco_annotations = sorted(coco_annotations, key=lambda x: x['id'])
    coco_images = sorted(coco_images, key=lambda x: x['id'])
    js = {
        'images': coco_images,
        'categories': all_categories,
        'annotations': coco_annotations
    }

    with open(json_filename, 'w') as file:
        json.dump(js, file, indent=2, sort_keys=True)
    print("converting to COCO over!")

