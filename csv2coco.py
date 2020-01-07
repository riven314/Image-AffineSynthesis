"""
REFERENCE:
https://github.com/spytensor/prepare_detection_dataset/blob/master/csv2coco.py
"""
import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split
np.random.seed(41)

# 0 is background
classname_to_id = {"c": 1}

class Csv2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  

    # from txt to COCO format
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi,label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # init COCO image dict
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(os.path.join(self.image_dir, path), cv2.IMREAD_GRAYSCALE)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # init COCO annotation dict
    def _annotation(self, shape,label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        print('classname_to_id: {}'.format(classname_to_id))
        print('label: {}'.format(label))
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # box format: [x1, y1, w, h]
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # calc area
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        assert max_x != min_x, '[ERROR] max_x cant be equal to min_x'
        assert max_y != min_y, '[ERROR] max_y cant be equal to min_y'
        return (max_x - min_x) * (max_y - min_y)
    
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x, min_y, min_x, min_y + 0.5*h, min_x, max_y,
                  min_x + 0.5 * w, max_y, max_x, max_y, max_x, max_y - 0.5 * h, 
                  max_x, min_y, max_x - 0.5 * w, min_y])
        return a
   

if __name__ == '__main__':
    # preprocessing
    import argparse
    parser = argparse.ArgumentParser(description = 'convert csv-format dataset to VOC-format dataset')
    parser.add_argument('--csv_base_dir', help = 'base dir to csv format data, the csv folder', type = str)
    parser.add_argument('--coco_base_dir', help = 'base dir to COCO format data, this dir build coco_', type = str)
    parser.add_argument('--test_fraction', help = 'how much portion for validation set (0.-1.)', type = float, default = 0.2)
    args = parser.parse_args()

    csv_base_dir = args.csv_base_dir
    csv_file = os.path.join(csv_base_dir, 'labels.csv')
    image_dir = os.path.join(csv_base_dir, 'images')
    saved_coco_path = args.coco_base_dir
    test_fraction = args.test_fraction

    assert os.path.isdir(csv_base_dir), '[ERROR] csv data base dir not exist: {}'.format(csv_base_dir)
    assert os.path.isdir(saved_coco_path), '[ERROR] COCO data base dir not exist: {}'.format(saved_coco_path)
    
    # collect csv format .xml files
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file,header=None).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
        else:
            total_csv_annotations[key] = value

    # sort train, val split
    inst_cnt_set = set([i[:5] for i in total_csv_annotations.keys()])
    train_keys, val_keys = [], []
    for inst_cnt in inst_cnt_set:
        target_files = [i for i in total_csv_annotations.keys() if inst_cnt in i]
        tmp_train_files, tmp_val_files = train_test_split(target_files, 
                                                          test_size = test_fraction, 
                                                          random_state = 42)
        train_keys.extend(tmp_train_files)
        val_keys.extend(tmp_val_files)

    print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    # init COCO folder structure
    if not os.path.exists(os.path.join(saved_coco_path, 'coco', 'annotations')):
        os.makedirs(os.path.join(saved_coco_path, 'coco', 'annotations'))
    if not os.path.exists(os.path.join(saved_coco_path, 'coco', 'images', 'train2017')):
        os.makedirs(os.path.join(saved_coco_path, 'coco', 'images', 'train2017'))
    if not os.path.exists(os.path.join(saved_coco_path, 'coco', 'images', 'val2017')):
        os.makedirs(os.path.join(saved_coco_path, 'coco', 'images', 'val2017'))
    # convert training data into COCO format
    l2c_train = Csv2CoCo(image_dir = image_dir, total_annos = total_csv_annotations)
    train_instance = l2c_train.to_coco(train_keys)
    l2c_train.save_coco_json(train_instance, os.path.join(saved_coco_path, 'coco', 'annotations', 'instances_train2017.json'))
    for file in train_keys:
        shutil.copy(os.path.join(image_dir, file), os.path.join(saved_coco_path, 'coco', 'images', 'train2017'))
    for file in val_keys:
        shutil.copy(os.path.join(image_dir, file), os.path.join(saved_coco_path, 'coco', 'images', 'val2017'))
    # convert val data into COCO format
    l2c_val = Csv2CoCo(image_dir = image_dir, total_annos = total_csv_annotations)
    val_instance = l2c_val.to_coco(val_keys)
    l2c_val.save_coco_json(val_instance, os.path.join(saved_coco_path, 'coco', 'annotations', 'instances_val2017.json'))