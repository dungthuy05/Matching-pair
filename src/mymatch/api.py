import numpy as np
from .detectors import detect_and_describe_sift
from .matchers import match_descriptors_knn
from .filters import ransac_filter


def get_matching_pairs(img1, img2, ratio=0.75, ransac_thresh=1.0):
    pts1, desc1 = detect_and_describe_sift(img1)
    pts2, desc2 = detect_and_describe_sift(img2)
    if desc1 is None or desc2 is None or len(pts1) == 0 or len(pts2) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), {"inliers": 0}

    matches = match_descriptors_knn(desc1, desc2, ratio)
    if matches.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0, 2)), {"inliers": 0}

    src = pts1[matches[:, 0]]
    dst = pts2[matches[:, 1]]

    src_f, dst_f, mask, F = ransac_filter(src, dst, ransac_thresh)
    info = {"inliers": int(mask.sum()), "F": F}
    return src_f, dst_f, info
