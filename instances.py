"""
high-level implementation to generate instances, with random geometric transformation

class object design
    input:
        - dir to the crop chromosome images and their masks
        - # chromosomes to be generated
    output:
        - dict of chromosome instances and its masks
        - aggregate mask
        - mask on overlapping areas only
"""
import os
import sys
from random import sample

import cv2
import numpy as np
from numpy.random import choice

from sampling import sample_transform_matrix, sample_color_contrast_param
from mask import is_ofb, is_serious_overlap, is_double_overlap, is_big_relative_overlap, is_multi_cross
from mask import update_overlap_mask
from pair import apply_geo_transform, update_agg_inst_n_mask
from color import make_color_dull


class Instances:
    """
    class object for generating image and mask of a bunch of overlapping instances (WITHOUT background effect)
    random sample instance images and mask images from a dir and do geometric transformation on them
    
    init attributes:
        inst_dir -- dir to the instance images and mask images
        size -- size of output image

    computing attributes:
        inst_dict -- each key maps to (instance_image, instance_mask)
        agg_mask -- union of masks for current aggregated instances
        overlap_mask -- bookmark overlapping area for current instances
        agg_inst -- image aggregated instances (WITHOUT background)

    """
    def __init__(self, img_dir, size):
        self.img_dir = img_dir
        self.size = size
        self.get_tif_ls()
        # bookmark tif being used in an image
        self.crt_tif = None 
        self.used_tif = []

    def propose_valid_img(self, inst_n, is_change_color = True):
        """
        propose n valid instance, main method to be used

        input:
            inst_n -- int, number of instances to be proposed
            size -- tuple, size of the instance and mask image
        output:
            out_dict -- dict, wrap all results into different keys
                - key: inst_dict -- each key is {'image': instance image, 'mask': mask} (transformation applied)
                - key: agg_inst -- np uint8 array, aggregated instances image
                - key: agg_mask -- np uint8 array, mask of aggregated instances
                - key: overlap_mask -- np uint8 array, bookmark aggregated overlapping area
        """
        # init inst_dict, agg_mask, overlap_mask
        inst_dict = {}
        agg_inst = np.zeros(shape = self.size, dtype = np.uint8) + 255
        agg_mask = np.zeros(shape = self.size, dtype = np.uint8)
        overlap_mask = np.zeros(shape = self.size, dtype = np.uint8)
        # sample parameters for color contrast
        alpha, constant = sample_color_contrast_param()
        # generate inst_n valid instances
        for i in range(inst_n):
            # avoid same tif file to be sampled again
            propose_inst, propose_mask = self.propose_a_valid_inst(agg_mask, overlap_mask)
            if is_change_color:
                propose_inst = make_color_dull(propose_inst, propose_mask, alpha, constant)
            bbox_ls = self.get_bbox(propose_mask)
            inst_dict[i + 1] = {'image': propose_inst, 'mask': propose_mask, 'bbox': bbox_ls}
            # must do update_overlap_mask BEFORE update_agg_mask
            overlap_mask = update_overlap_mask(propose_mask, agg_mask, overlap_mask)
            agg_inst, agg_mask = update_agg_inst_n_mask(propose_inst, propose_mask, agg_inst, agg_mask)
        # sanity check and then clean self.crt_tif, self.used_tif
        assert len(self.used_tif) == inst_n, '[ERROR] used_tif len != inst_n'
        self.crt_tif = None
        self.used_tif = []
        # wrap all results into a dict
        out_dict = {} 
        out_dict['inst_dict'] =  inst_dict
        out_dict['agg_inst'] = agg_inst
        out_dict['agg_mask'] = agg_mask
        out_dict['overlap_mask'] = overlap_mask
        out_dict['color_param'] = [alpha, constant]
        return out_dict

    def get_bbox(self, mask, delta = 2):
        """
        bounding box is parameterised by (x_min, y_min, x_max, y_max)

        output: 
            bbox_ls -- list, [x_min, y_min, x_max, y_max]
        """
        h, w = mask.shape
        xs, ys = np.where(mask != 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_min = max(x_min - delta, 0)
        y_min = max(y_min - delta, 0)
        x_max = min(x_max + delta, h)
        y_max = min(y_max + delta, w)
        return [x_min, y_min, x_max, y_max]

    def propose_a_valid_inst(self, agg_mask, overlap_mask):
        """
        given current aggregated mask and mask on overlapping areas. 
        propose a valid mask, tgt with the corresponding instance (with geometric transformation)

        input:
            agg_mask -- uint8 np array, aggregated mask of current k masks
            overlap_mask -- uint8 np array, book mark of overlapping areas of current k instances
        output:
            propose_inst, propose_mask -- uint8 np array, valid instance and its mask
        """
        propose_inst, propose_mask = self.sample_trans_inst()
        # reject either if the propose mask is invalid or the tif file is already used
        while (not self.is_valid_mask(propose_mask, agg_mask, overlap_mask)) or (self.crt_tif in self.used_tif):
            propose_inst, propose_mask = self.sample_trans_inst()
        self.used_tif.append(self.crt_tif)
        return propose_inst, propose_mask

    def sample_trans_inst(self):
        """
        sample (instance image, instance mask) pair and then sample geometric transformation on them
        """
        one_inst, one_mask = self.sample_inst()
        trans_mat = sample_transform_matrix(self.size)
        propose_inst, propose_mask = apply_geo_transform(one_inst, one_mask, trans_mat, self.size)
        return propose_inst, propose_mask

    def sample_inst(self):
        """
        sample (instance image, instance mask) pair from img_dir
        """
        # sample .tif, .png pair
        #tif_f = sample(self.tif_ls, 1)[0]
        tif_f = choice(self.tif_ls, size = 1, replace = False)[0]
        png_f = '{}.png'.format(os.path.splitext(tif_f)[0])
        # construct path to .tif and .png and check existance
        tif_path = os.path.join(self.img_dir, tif_f)
        png_path = os.path.join(self.img_dir, png_f)
        assert os.path.isfile(tif_path), '[ERROR] .tif not exist: {}'.format(tif_path)
        assert os.path.isfile(png_path), '[ERROR] .png not exist: {}'.format(png_path)
        # read .tif and .png into numpy array
        one_inst = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
        one_mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        # bookmark current tif being sampled
        self.crt_tif = tif_f
        return one_inst, one_mask

    def is_valid_mask(self, one_mask, agg_mask, overlap_mask):
        is_valid = False
        if is_ofb(one_mask):
            print('rejected because out of box')
        elif is_serious_overlap(agg_mask, one_mask):
            print('rejected because big overlap area')
        elif is_double_overlap(overlap_mask, one_mask):
            print('rejected because wide overlap area')
        elif is_multi_cross(agg_mask, one_mask):
            print('rejected because one_mask overlap so many times')
        elif is_big_relative_overlap(agg_mask, one_mask):
            print('rejected because one_mask overlap in relative big area')
        # pass the test if the above pass
        else:
            is_valid = True
        return is_valid

    def get_tif_ls(self):
        assert self.img_dir is not None, '[ERROR] self.img_dir is None'
        assert os.path.isdir(self.img_dir), '[ERROR] self.img_dir doesnt exist'
        self.tif_ls = [f for f in os.listdir(self.img_dir) if f.endswith('tif')]
        assert len(self.tif_ls) != 0, '[ERROR] self.img_dir has no .tif instance image'


if __name__ == '__main__':
    pass
