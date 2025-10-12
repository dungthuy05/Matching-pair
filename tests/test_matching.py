import os
import cv2
from mymatch.api import get_matching_pairs

def test_demo_pair():
    here = os.path.dirname(__file__)
    ex1 = os.path.join(here, '..', 'examples', 'sample1.jpg')
    ex2 = os.path.join(here, '..', 'examples', 'sample2.jpg')
    img1 = cv2.imread(ex1)
    img2 = cv2.imread(ex2)
    pts1, pts2, info = get_matching_pairs(img1, img2)
    assert 'inliers' in info
