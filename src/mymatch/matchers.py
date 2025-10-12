import cv2
import numpy as np

def match_descriptors_knn(desc1, desc2, ratio=0.75):
    if desc1 is None or desc2 is None:
        return np.empty((0,2), dtype=int)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2: 
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    if len(good) == 0:
        return np.empty((0,2), dtype=int)
    return np.array(good, dtype=int)
