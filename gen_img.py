"""
aggregate several layers to generate an image, init by sampling parameters
- instance group (tgt with color change for reducing color contrast)
- background (can use SinGAN to generate that)
- (add air bubble / extra noise)
- (apply gaussian blur on overall image)

** retrieve bounding box ground truth
"""
import os
import sys

import cv2
import numpy as np

from instances import Instances
from background import generate_background


def gen_one_img(instances, inst_n, is_blur = True):
    """
    background merge with instances, output a dict
    """
    out_size = instances.size
    spotlight_size = (out_size[0] * 2, out_size[1] * 2)
    inst_res = instances.propose_valid_img(inst_n = inst_n)
    bg_img = generate_background(spotlight_size = spotlight_size,
                                 out_size = out_size)
    final_res = {}
    merge_img = merge_bg_inst(bg_img, 
                              inst_res['agg_inst'], 
                              inst_res['agg_mask'])
    if is_blur:
        merge_img = cv2.GaussianBlur(merge_img, (5, 5), 0)
    final_res['image'] = merge_img
    final_res['agg_mask'] = inst_res['agg_mask']
    final_res['overlap_mask'] = inst_res['overlap_mask']
    final_res['inst_dict'] = inst_res['inst_dict']
    return final_res


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

