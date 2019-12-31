"""
reference from: https://github.com/spytensor/prepare_detection_dataset/blob/master/csv2voc.py

TBD:
1. do equal splitting per inst_n
"""
import os
import argparse
from glob import glob

import shutil
import codecs
import json
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from IPython import embed

parser = argparse.ArgumentParser(description = 'convert csv-format dataset to VOC-format dataset')
parser.add_argument('--csv_base_dir', help = 'base dir to csv dataset (up till csv folder)', type = str)
parser.add_argument('--voc_base_dir', help = 'base dir to VOC dataset (VOCdevkit + VOC2007 is created in this dir)', type = str)
parser.add_argument('--test_fraction', help = 'how much portion for validation set (0.-1.)', type = float, default = 0.2)
args = parser.parse_args()

# argument construction
test_fraction = args.test_fraction
csv_base_dir = args.csv_base_dir
csv_file = os.path.join(csv_base_dir, 'labels.csv')
voc_base_dir = args.voc_base_dir
# create VOCdevkit\VOC2007 inside voc_base_dir
assert os.path.isdir(voc_base_dir), '[ERROR] VOC base dir not exist'
if not os.path.isdir(os.path.join(voc_base_dir, 'VOCdevkit')):
    os.mkdir(os.path.join(voc_base_dir, 'VOCdevkit'))
if not os.path.isdir(os.path.join(voc_base_dir, 'VOCdevkit', 'VOC2007')):
    os.mkdir(os.path.join(voc_base_dir, 'VOCdevkit', 'VOC2007'))
saved_path = os.path.join(os.path.join(voc_base_dir, 'VOCdevkit', 'VOC2007'))
image_save_path = "JPEGImages"
image_raw_path = os.path.join(csv_base_dir, 'images')

assert os.path.isdir(image_raw_path), '[ERROR] csv images dir not exist'
assert os.path.isfile(csv_file), '[ERROR] csv file does not exist'
print('csv base dir: {}'.format(csv_base_dir))
print('voc_base_dir: {}'.format(voc_base_dir))
print('test_fraction: {}'.format(test_fraction))


# init dirs for VOC folders
if not os.path.isdir(os.path.join(saved_path, 'Annotations')): 
    os.mkdir(os.path.join(saved_path, 'Annotations'))
if not os.path.isdir(os.path.join(saved_path, 'JPEGImages')): 
    os.mkdir(os.path.join(saved_path, 'JPEGImages'))
if not os.path.isdir(os.path.join(saved_path, 'ImageSets')):
    os.mkdir(os.path.join(saved_path, 'ImageSets'))
if not os.path.isdir(os.path.join(saved_path, 'ImageSets', 'Main')): 
    os.mkdir(os.path.join(saved_path, 'ImageSets', 'Main'))
    
# retrieve doc
total_csv_annotations = {}
annotations = pd.read_csv(csv_file, header = None).values
for annotation in annotations:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
    else:
        total_csv_annotations[key] = value

# annotation on xml files
for filename,label in total_csv_annotations.items():
    #embed() 
    height, width, channels = cv2.imread(os.path.join(image_raw_path, filename)).shape
    #embed()
    xml_path = os.path.join(saved_path, 'Annotations', filename.replace('.tif', '.xml'))
    with codecs.open(xml_path, "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'Karyotype' + '</folder>\n')
        xml.write('\t<filename>' + filename + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>Simulated Karyotype</database>\n')
        xml.write('\t\t<annotation>Chromosome Annotations</annotation>\n')
        xml.write('\t\t<image>biomedical</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>Alex Lau</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        if isinstance(label, float):
            ## blank
            xml.write('</annotation>')
            continue
        for label_detail in label:
            labels = label_detail
            #embed()
            xmin = int(labels[0])
            ymin = int(labels[1])
            xmax = int(labels[2])
            ymax = int(labels[3])
            label_ = labels[-1]
            assert xmin < xmax, '[BBOX ERROR] xmin >= xmax'
            assert ymin < ymax, '[BBOX ERROR] ymin >= ymax'
            xml.write('\t<object>\n')
            xml.write('\t\t<name>'+label_+'</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>1</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
            print(filename,xmin,ymin,xmax,ymax,labels)
        xml.write('</annotation>')
        

#6.split files for txt
txtsavepath = os.path.join(saved_path, 'ImageSets', 'Main/')
ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')
total_files = glob(os.path.join(saved_path, 'Annotations' , '*.xml'))
total_files = [os.path.splitext(os.path.split(i)[1])[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    ftrainval.write(file + '\n')

# copy tif. images to voc JPEGImages folder
for image in glob(os.path.join(image_raw_path, '*.tif')):
    shutil.copy(image, os.path.join(saved_path, image_save_path))

# TBD: better do even splitting per inst_n
inst_cnt_set = set([i[:5] for i in total_files])
train_files, val_files = [], []
for inst_cnt in inst_cnt_set:
    target_files = [i for i in total_files if inst_cnt in i]
    tmp_train_files, tmp_val_files = train_test_split(target_files, 
                                                      test_size = test_fraction, 
                                                      random_state = 42)
    train_files.extend(tmp_train_files)
    val_files.extend(tmp_val_files)

for file in train_files:
    ftrain.write(file + '\n')
#val
for file in val_files:
    fval.write(file + '\n')

ftrainval.close()
ftrain.close()
fval.close()