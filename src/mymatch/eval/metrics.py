import numpy as np


def precision_at_threshold(pts1, pts2, gt_matches=None, thresh=3.0):
    if len(pts1) == 0:
        return 0.0
    d = np.linalg.norm(pts1 - pts2, axis=1)
    return float((d <= thresh).sum()) / len(d)
