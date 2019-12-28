"""
low-level implementation between masks
"""
import numpy as np
import cv2


def update_agg_mask(one_mask , agg_mask):
    """
    superimpose one_mask on agg_mask to update agg_mask
    change agg_mask in-place
    """
    agg_mask = np.logical_or(one_mask, agg_mask)
    agg_mask = np.uint8(agg_mask)
    agg_mask[agg_mask != 0] = 255
    return agg_mask


def update_overlap_mask(one_mask, agg_mask, overlap_mask):
    """
    bookmark new overlap area (between one_mask and agg_mask) into overlap_mask
    change overlap_mask in-place
    """
    intersect_mask = np.logical_and(one_mask, agg_mask)
    xs, ys = np.where(intersect_mask)
    overlap_mask[xs, ys] = 255
    return overlap_mask


def is_ofb(one_mask, threshold = 400):
    """
    check if the transform instance is completely / almost out of box
    ** it may mistakenly reject small instance that is not out of box

    input:
        one_mask -- uint8 np array, mask after geometric transform
        threshold -- int, number of instance pixels for tolerance
    output:
        boolean, True: transform instance is out of box
    """
    xs = np.where(one_mask != 0)[0]
    if len(xs) <= threshold:
        return True
    else:
        return False


def is_serious_overlap(agg_mask, one_mask, area_threshold = 250, length_threshold = 30):
    """
    check two mask overlap with an area too large / too wide. area should be bounded by square
    check any disjoint set of overlapping areas

    input:
        agg_mask -- uint8 np array, first k instances aggregated masks
        one_mask -- uint8 np array, mask to be proposed into aggregation
    output:
        boolean -- True: having a serious overlapping
    """
    intersect_mask = np.logical_and(agg_mask, one_mask)
    # change from bool to np.uint8 before cv2.connectedComponents
    intersect_mask = np.uint8(intersect_mask)
    _, labels = cv2.connectedComponents(intersect_mask)
    idx_ls = np.unique(labels)
    for idx in idx_ls:
        # omit background
        if idx == 0:
            continue
        n = (labels == idx).sum()
        # check for big overlapping area
        if n >= area_threshold:
            return True
        xs, ys = np.where(labels == idx)
        h = xs.max() - xs.min()
        w = ys.max() - ys.min()
        # check for wide overlapping area
        if max(h, w) >= length_threshold:
            return True
    return False


def is_double_overlap(overlap_mask, one_mask, threshold = 20):
    """
    check no more than 2 instances overlap on same area

    input:
        overlap_mask -- uint8 np array, mask bookmarking overlapping areas
        one_mask -- uint8 np array, mask to be proposed into aggregation
    output:
        boolean -- True: having double overlapping
    """
    intersect_mask = np.logical_and(overlap_mask, one_mask)
    if intersect_mask.sum() >= threshold:
        return True
    else:
        return False