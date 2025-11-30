import os

import cv2
import numpy as np
import torch

from .detectors import detect_and_describe_sift
from .filters import ransac_filter
from .matchers import match_descriptors_knn
from .transformer_matcher import TransformerMatcher

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tf_matcher = TransformerMatcher(
    desc_dim=128, d_model=256, nhead=4, ff_ratio=4, device=_device
)


def get_matching_pairs(
    img1,
    img2,
    mask1_path=None,
    mask2_path=None,
    ratio=0.75,
    ransac_thresh=1.0,
    use_transformer: bool = True,
):
    """
    High-level API: get matching pairs between img1 and img2.
    Returns: pts1 (N,2), pts2 (N,2), info (dict)
    info gồm: inliers, F, matches_before_ransac, matcher
    """
    # 1) Áp mask
    if mask1_path and os.path.exists(mask1_path):
        mask1 = cv2.imread(mask1_path, 0)
        img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    if mask2_path and os.path.exists(mask2_path):
        mask2 = cv2.imread(mask2_path, 0)
        img2 = cv2.bitwise_and(img2, img2, mask=mask2)

    # 2) SIFT keypoints + descriptors
    pts1, desc1 = detect_and_describe_sift(img1)
    pts2, desc2 = detect_and_describe_sift(img2)

    if desc1 is None or desc2 is None or len(pts1) == 0 or len(pts2) == 0:
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            {
                "inliers": 0,
                "matches_before_ransac": 0,
                "matcher": "none",
                "F": None,
            },
        )

    # 3) Lấy matches
    if use_transformer:
        matches_tf = _tf_matcher(
            pts1, desc1, pts2, desc2, mutual_check=True, min_score=0.0
        )
        # matches_tf: (M,3) => (idx1, idx2, score)
        if matches_tf.shape[0] == 0:
            return (
                np.empty((0, 2)),
                np.empty((0, 2)),
                {
                    "inliers": 0,
                    "matches_before_ransac": 0,
                    "matcher": "transformer",
                    "F": None,
                },
            )
        idx1 = matches_tf[:, 0].astype(int)
        idx2 = matches_tf[:, 1].astype(int)
        matches = np.stack([idx1, idx2], axis=1).astype(int)
        matcher_name = "transformer"
    else:
        matches = match_descriptors_knn(desc1, desc2, ratio)
        if matches.shape[0] == 0:
            return (
                np.empty((0, 2)),
                np.empty((0, 2)),
                {
                    "inliers": 0,
                    "matches_before_ransac": 0,
                    "matcher": "knn",
                    "F": None,
                },
            )
        matcher_name = "knn"

    # 4) Chuẩn bị src, dst cho RANSAC
    src = pts1[matches[:, 0]]
    dst = pts2[matches[:, 1]]

    # 5) Lọc bằng RANSAC Fundamental
    src_f, dst_f, mask, F = ransac_filter(src, dst, ransac_thresh)
    info = {
        "inliers": int(mask.sum()),
        "F": F,
        "matches_before_ransac": matches.shape[0],
        "matcher": matcher_name,
        "src_all": src,  # (M,2)
        "dst_all": dst,  # (M,2)
        "mask_inliers": mask,  # (M,)
    }
    return src_f, dst_f, info
