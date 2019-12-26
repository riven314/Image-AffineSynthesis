"""
low-level implementation to change color contrast of the instances
smooth out the white pixel artifact at boundary
"""
import numpy as np

def make_color_dull(inst_img, inst_mask, alpha, constant):
    """
    reduce color contrast (look duller) of inst_img, change in-place
    make the instance color dull by the following formula:
    - new_intensity = alpha * old_intensity + constant

    input:
        inst_img -- uint8 np array, instance image after geometric transform
        inst_mask -- uint8 np array, mask after same geometric transform
        alpha -- float (0-1), control the scale of current intensity 
    """
    inst_img[inst_mask != 0] = inst_img[inst_mask != 0] * alpha + constant
    return inst_img


def smooth_boundary(inst_img, inst_mask):
    """
    instance image may have artifact (white pixel) near boundary
    smoothen the pixels by nearest neighbor
    """
    pass
    
