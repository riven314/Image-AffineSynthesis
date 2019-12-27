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

import cv2
import numpy as np

from sampling import sample_transform_matrix
from mask import is_ofb, is_serious_overlap, is_double_overlap, update_overlap_mask, update_agg_inst_n_mask
from pair import apply_geo_transform


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

    def propose_n_valid_inst(self, inst_n, size):
        """
        propose n valid instance

        input:
            inst_n -- int, number of instances to be proposed
            size -- tuple, size of the instance and mask image
        """
        # init inst_dict, agg_mask, overlap_mask
        inst_dict = {}
        agg_mask = np.zeros(shape = size, dtype = np.uint8)
        overlap_mask = np.zeros(shape = size, dtype = np.uint8)
        # generate inst_n valid instances
        for i in range(inst_n):
            propose_inst, propose_mask = self.propose_a_valid_inst(agg_mask, overlap_mask)
            inst_dict[i + 1] = {'image': propose_inst, 'mask': propose_mask}
            # must do update_overlap_mask BEFORE update_agg_mask
            overlap_mask = update_overlap_mask(propose_mask, agg_mask, overlap_mask)
            agg_inst, agg_mask = update_agg_inst_n_mask(propose_inst, propose_mask, agg_inst, agg_mask)
        # set as class object properties
        self.inst_dict = inst_dict
        self.agg_inst = agg_inst
        self.agg_mask = agg_mask
        self.overlap_mask = overlap_mask

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
        while not self.is_valid_mask(propose_mask, agg_mask, overlap_mask):
            propose_inst, propose_mask = self.sample_trans_inst()
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
        one_inst, one_mask = None, None
        return one_inst, one_mask

    def is_valid_mask(self, one_mask, agg_mask, overlap_mask):
        if is_ofb(one_mask):
            print('rejected because out of box')
            return False
        elif is_serious_overlap(agg_mask, one_mask):
            print('rejected because big overlap area')
            return False
        elif is_double_overlap(overlap_mask, one_mask):
            print('rejected because wide overlap area')
            return False
        # pass the test if the above pass
        else:
            return True


if __name__ == '__main__':
    pass
