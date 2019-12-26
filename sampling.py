"""
low-level implementation to do sampling on:
1. number of instances
2. parameters on geometric transformation (translation and rotation)
3. background paramters

all subject to user define distributions
"""
import os
import sys
from math import pi
from random import sample

import cv2
import numpy as np


def sample_file(files_dir):
    """
    sample .tif image file and .png mask file pair
    
    input: file_dir -- dir to .tif images and .png masks 
    """
    tif_ls = [f for f in os.listdir(files_dir) if f.endswith('.tif')]
    png_ls = [f for f in os.listdir(files_dir) if f.endswith('.png')]
    pairs_ls = [(tif, png) for tif, png in zip(tif_ls, png_ls)]
    sample_pair = sample(pairs_ls, 1)[0]
    sample_pair = (os.path.join(files_dir, sample_pair[0]), os.path.join(files_dir, sample_pair[1]))
    return sample_pair


def sample_transform_matrix(size):
    h, w = size
    loc = sample_translate_param(size, l_limit = -70, h_limit = h - 160 + 80)
    rad_fraction = sample_rotate_param()
    trans_mat = get_transform_matrix(loc, rad_fraction)
    return trans_mat


def sample_translate_param(size, l_limit, h_limit):
    """
    random sample translation parameter (a tuple)
    """
    loc = np.random.uniform(low = l_limit, high = h_limit, size = 2)
    return loc


def sample_rotate_param():
    """
    random sample rotation parameter (0*pi - 2*pi)
    """
    rad_fraction = np.random.uniform(low = 0, high = 2)
    return rad_fraction


def get_transform_matrix(loc, rad_fraction):
    """
    rotation_matrix * translation_matrix, both 3x3 matrix
    at the end, truncate it back to 2x3 matrix
    """
    rotation_mat = get_rotate_matrix(rad_fraction)
    translate_mat = get_translate_matrix(loc)
    # pad 2x3 matrices into 3x3
    rotation_mat = np.vstack([rotation_mat, [0, 0, 1]])
    translate_mat = np.vstack([translate_mat, [0, 0, 1]])
    # first apply rotation (at instance center), then translation
    # truncate back to 2x3 matrix after multiplication
    out_mat = (translate_mat @ rotation_mat)[:2]
    return out_mat


def get_rotate_matrix(rad_fraction, center = (80, 80)):
    """
    default rotation center to be (80, 80) as image size is (160, 160)

    input:
        rad_fraction -- float, 0-2 (i.e. 0 rad - 2pi rad)
        center -- tuple, center of rotation, default to be center of instance
    output:
        rotate_mat -- 2x3 np matrix for rotation
    """
    rotate_mat = cv2.getRotationMatrix2D(center, np.rad2deg(rad_fraction * pi), 1)
    return rotate_mat


def get_translate_matrix(loc):
    """
    input:
        loc -- tuple (tx, ty) -- tx refers to vertical change, ty refers to horizontal
    output:
        translate_mat -- 2x3 np matrix for translation
    """
    tx, ty = loc
    translate_mat = np.array(
        [[1, 0, ty], [0, 1, tx]], # note that ty comes first
        dtype = np.float64)
    return translate_mat
