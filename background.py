import os
import sys

import cv2
import  numpy as np


def gen_background(mean, sd, size):
    """
    input:
        mean -- mean int8 intensity of background
        sd -- std dev of background gaussian noise
        size -- tupe (h,w), size of background
    output:
        bg -- (h, w) np array, (h, w) = size
    """
    bg_img = np.random.normal(mean, sd, size)
    bg_img = np.uint8(bg_img)
    return bg_img


def merge_bg_inst(bg_img, inst_img, inst_mask):
    """
    input:
        bg_img -- (h, w) np array, original background image (without any masking)
        inst_img -- (h, w) np array, original instance groups image (without any masking)
        inst_mask -- (h, w) np array, instance groups mask (255 as white, 0 as black)
    output:
        merge_img -- (h, w) np array, instances group merged with background 
    * some artifact on boundary side may remain 
    """
    masked_bg = np.logical_not(inst_mask) * bg_img
    masked_fg = (inst_mask != 0) * inst_img
    return masked_bg + masked_fg