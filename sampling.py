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


def sample_background_param():
    mean = np.random.uniform(low = 150., high = 230.)
    sd = np.random.uniform(low = 0, high = 20)
    return mean, sd


def sample_spotlight_param(img_size, crop_size):
    """
    input:
        img_size -- tuple, size of image to be cropped
        crop_size -- tuple, size of crop image
    """
    h, w = img_size
    crop_h, crop_w = crop_size
    out_h = np.random.randint(low = 0, high = h - crop_h)
    out_w = np.random.randint(low = 0, high = w - crop_w)
    light_mean = np.random.uniform(20, 80)
    light_sd = np.random.uniform(60, 120)
    return (out_h, out_w), light_mean, light_sd


def sample_color_contrast_param():
    """
    random sample parameters for making color contrast on instances

    output:
        alpha -- float, (0.8 - 1.0)
        constant -- uniform, (20 - 60)*-1
    """
    alpha = np.random.uniform(low = 0.75, high = 1.)
    constant = np.random.uniform(low = 20, high = 60) * -1
    return alpha, constant
    

def sample_transform_matrix(size, loc = None, rad_fraction = None):
    """
    randomly sample a geometric transformation matrix if loc, rad_fraction not specified

    input:
        size -- tuple, size of output image
        loc -- tuple (tx, ty), translation parameters. tx - vertical, ty - horizontal
        rad_fraction -- float (0-2), fraction of 2*pi
    """
    h, w = size
    if loc is None:
        loc = sample_translate_param(size, l_limit = -70, h_limit = h - 160 + 80)
    if rad_fraction is None:
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
