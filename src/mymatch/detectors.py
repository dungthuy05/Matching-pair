import cv2
import numpy as np


def detect_and_describe_sift(img, mask=None, detectorParams=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if detectorParams is None:
        detectorParams = {}
    sift = cv2.SIFT_create(**detectorParams)
    kps, desc = sift.detectAndCompute(gray, mask)
    if not kps:
        return np.empty((0, 2), dtype=np.float32), None
    pts = np.array([kp.pt for kp in kps], np.float32)
    return pts, desc
