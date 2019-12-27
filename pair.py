"""
low-level implementation for most operations acting on (instance img, mask) pair
"""
import cv2
import numpy as np
from mask import update_agg_mask


def apply_geo_transform(one_img, one_mask, trans_mat, size):
    """
    apply the same geometric transform on instance image and the mask
    change in-place

    input:
        one_img -- uint8 np array, image of the instance (shd be 160 x 160)
        one_mask -- uint8 np array, mask of the instance (shd be 160 x 160)
        trans_mat -- 2x3 np array, transformation matrix
    output:
        new_img, new_mask -- uint8 np array, instance image and mask after transform
    """
    new_img = cv2.warpAffine(one_img, trans_mat, size, 
                             flags = cv2.INTER_NEAREST,  # this setting is important for preserve original color space
                             borderMode = cv2.BORDER_CONSTANT, 
                             borderValue = 255) # white background for instance image
    new_mask = cv2.warpAffine(one_mask, trans_mat, size, 
                             flags = cv2.INTER_NEAREST,  
                             borderMode = cv2.BORDER_CONSTANT, 
                             borderValue = 0) # black background for mask
    return new_img, new_mask


def update_agg_inst_n_mask(one_inst, one_mask, agg_inst, agg_mask):
    agg_inst = update_agg_inst(one_inst, one_mask, agg_inst)
    agg_mask = update_agg_mask(one_mask, agg_mask)
    return agg_inst, agg_mask


def update_agg_inst(one_inst, one_mask, agg_inst):
    """
    superimpose one_inst on agg_inst to update agg_inst
    change agg_inst in-place
    """
    agg_inst = (one_mask == 0) * agg_inst + (one_mask != 0) * one_inst
    return agg_inst