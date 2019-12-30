"""
generate image by gen_img.py
and construct as csv format, demonstrated as follows:

*** structure ***
csv/
    labels.csv
    images/
        image1.jpg
        image2.jpg
        ...

*** csv format ***
/path/to/image,xmin,ymin,xmax,ymax,label
e.g. /mfs/dataset/face/0d4c5e4f-fc3c-4d5a-906c-105.jpg,450,154,754,341,face
(please use absolute path)
"""
import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from gen_img import gen_one_img
from instances import Instances


parser = argparse.ArgumentParser(description = 'simulate image, write and construct a csv file')
parser.add_argument('--crop_img_dir', help = 'path to crop image, mask pairs, can be relative path', type = str)
parser.add_argument('--data_base_dir', help = 'base dir for writing image and csv file, MUST BE absolute path', type = str)
parser.add_argument('--out_len', help = 'a side of simulated image, assume square', type = int, default = 320)
parser.add_argument('--img_per_n', help = '# simulated images per inst_n', type = int)
parser.add_argument('--max_inst_n', help = 'max. # instances to be put in an image', type = int, default = 25)
args = parser.parse_args()


def build_csv_dataset(crop_img_dir, data_base_dir, out_size, max_inst_n, img_per_n):
    """
    simulate images and store inside data_base_dir. data format is in csv

    input:
        crop_img_dir -- path to the crop image, mask pairs
        data_base_dir -- path for building dataset
        out_size -- tuple, (h, w) of simulated image
        max_inst_n -- int, max # instances you can put in an image
        img_per_n -- int, # simulated images per inst_n
    """
    # init write folder and csv writer
    if not os.path.isdir(data_base_dir):
        os.mkdir(data_base_dir)
        print('dir created: {}'.format(data_base_dir))
    img_dir = os.path.join(data_base_dir, 'images')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
        print('dir created: {}'.format(img_dir))
    csv_writer = open(os.path.join(data_base_dir, 'labels.csv'), 'w')
    temp_img_f = 'cnt{}_no{}.tif'
    # init instances object
    inst = Instances(crop_img_dir, size = out_size)
    # iterate simulation, start from inst_n = 1
    for inst_n in range(1, max_inst_n + 1):
        print('instance #: {}'.format(inst_n))
        for i in range(max_inst_n):
            img_f = temp_img_f.format(inst_n, i+1)
            w_path = os.path.join(img_dir, img_f)
            result_dict = gen_one_img(inst, inst_n)
            write_csv(csv_writer, result_dict, w_path)
            cv2.imwrite(w_path, result_dict['image'])
            print('written: {}'.format(w_path))
    print('simulation completed!')


def write_csv(csv_writer, result_dict, w_path):
    """
    action: write [/path/to/image, xmin, ymin, xmax, ymax, label]

    input:
        csv_writer -- csv writer for writing files, open(..)
        result_dict -- output from gen_one_img
        w_path -- str, absolute path for writing images
    """
    for _, inst_dict in result_dict['inst_dict'].items():
        bbox_ls = inst_dict['bbox']
        bbox_ls = [int(i) for i in bbox_ls]
        s = ','.join(map(str, bbox_ls))
        s = w_path + ',' + s
        s = s + '\n'
        csv_writer.write(s)


if __name__ == '__main__':
    crop_img_dir = args.crop_img_dir
    data_base_dir = args.data_base_dir
    out_size = (args.out_len, args.out_len)
    img_per_n = args.img_per_n
    max_inst_n = args.max_inst_n
    build_csv_dataset(crop_img_dir, data_base_dir, out_size, max_inst_n, img_per_n)