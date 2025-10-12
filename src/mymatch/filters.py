import cv2
import numpy as np


def ransac_filter(pts1, pts2, ransac_thresh=1.0):
    if len(pts1) < 8:
        mask = np.zeros((len(pts1),), dtype=bool)
        return pts1, pts2, mask, None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, 0.99)
    mask = (
        mask.ravel().astype(bool)
        if mask is not None
        else np.zeros((len(pts1),), dtype=bool)
    )
    return pts1[mask], pts2[mask], mask, F
