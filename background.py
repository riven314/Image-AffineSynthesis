"""
sub-module for generating background:
- mask a circular mask: https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays/44874588

"""
import os
import sys

import cv2
import  numpy as np
from scipy import signal

from sampling import sample_background_param, sample_spotlight_param


def generate_background(spotlight_size, out_size):
    """
    plain background with pixelwise noise + spotlight effect

    input:
        spotlight_size -- tuple, size of spotlight mask (usually bigger e.g. 640)
        crop_size -- tuple, size of final output
    """
    bg_mean = sample_background(out_size)
    spotlight = sample_spotlight(spotlight_size, out_size)
    bg_img = merge_spotligh_bg(spotlight, bg_mean)
    bg_img = cv2.GaussianBlur(bg_img, (13, 13), 0)
    return bg_img


def sample_background(size):
    mean, sd = sample_background_param()
    bg_mean = get_background_mean(mean, sd, size)
    return bg_mean


def sample_spotlight(spotlight_size, crop_size):
    """
    input:
        spotlight_size -- tuple, size of spotlight mask (usually bigger e.g. 640)
        crop_size -- tuple, size of final output
    """
    loc, light_mean, light_sd = sample_spotlight_param(spotlight_size, crop_size)
    spotlight = get_spotlight(spotlight_size, light_mean, light_sd)
    crop_spotlight = crop_mask(spotlight, loc, crop_size)
    return crop_spotlight


def get_background_mean(mean, sd, size):
    """
    get plain background with pixel-wise gaussian noise

    input:
        mean -- mean int8 intensity of background
        sd -- std dev of background gaussian noise
        size -- tupe (h,w), size of background
    output:
        bg -- (h, w) np array, (h, w) = size
    """
    bg = np.zeros(shape = size) + mean
    noise = np.random.normal(0, sd, size)
    bg = np.clip(bg + noise, a_min = 0, a_max = 255)
    bg = bg.astype(np.uint8)
    return bg


def crop_mask(mask, loc, size):
    """
    input:
        mask -- np uint8 array
        loc -- tuple, (h, w) top left corner of crop image
        size -- tuple, size of the crop image
    """
    h, w = loc
    dh, dw = size
    crop_mask = mask[h:h + dh, w:w + dw]
    return crop_mask


def get_spotlight(size, mean, std):
    """
    use 2D Gaussian kernel to make circular spotlight background
    then, translate the spotlight to random location (loc)   

    input:
        size -- tuple, size of output mask
        mean -- mean intensity of spotlight
        std -- dispersion of intensity
        loc -- translation
    """
    kernel_l = size[0]
    gkern1d = signal.gaussian(kernel_l, std=std).reshape(kernel_l, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = np.clip(gkern2d * mean, a_min = 0, a_max = 255)
    gkern2d = gkern2d.astype(np.uint8)
    return gkern2d


def merge_spotligh_bg(spotlight, bg):
    bg_img = np.clip(spotlight.astype(np.int64) + bg.astype(np.int64), a_min = 0, a_max = 255)
    bg_img = bg_img.astype(np.uint8)
    return bg_img    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    big_size = (640, 640)
    size = (320, 320)
    bg = generate_background(big_size, size)
    plt.imshow(bg, cmap = 'gray', vmin = 0, vmax = 255)
    plt.show()
