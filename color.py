"""
low-level implementation to change color contrast of the instances
smooth out the white pixel artifact at boundary
"""
import numpy as np

def make_color_dull(inst_img, inst_mask, alpha, constant):
    """
    reduce color contrast (look duller) of inst_img
    make the instance color dull by the following formula:
    - new_intensity = alpha * old_intensity + constant
    - constant could be negative, do clipping to prevent intensity inversion

    input:
        inst_img -- uint8 np array, instance image after geometric transform
        inst_mask -- uint8 np array, mask after same geometric transform
        alpha -- float (0-1), control the scale of current intensity 
    """
    # create a new copy of inst_img with np.int8 (signed!)
    new_inst_img = inst_img.copy()
    new_inst_img.dtype = np.int8
    new_inst_img[inst_mask != 0] = np.clip(inst_img[inst_mask != 0] * alpha + constant,
                                      a_min = 0, a_max = None)
    # recover back from np.int8 to np.uint8
    new_inst_img = new_inst_img.astype(np.uint8)
    return new_inst_img


def smooth_boundary(inst_img, inst_mask):
    """
    instance image may have artifact (white pixel) near boundary
    smoothen the pixels by nearest neighbor
    """
    pass
