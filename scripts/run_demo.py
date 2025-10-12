import cv2, os
from mymatch.api import get_matching_pairs

here = os.path.dirname(__file__)
ex1 = os.path.join(here, "..", "examples", "sample1.jpg")
ex2 = os.path.join(here, "..", "examples", "sample2.jpg")

img1 = cv2.imread(ex1)
img2 = cv2.imread(ex2)
if img1 is None or img2 is None:
    raise RuntimeError("Example images not found.")

pts1, pts2, info = get_matching_pairs(img1, img2)
print("Inliers found:", info["inliers"])


def draw_simple_matches(img1, img2, pts1, pts2, max_draw=200):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = 255 * np.ones((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2
    for (x1, y1), (x2, y2) in zip(pts1[:max_draw], pts2[:max_draw]):
        cv2.circle(canvas, (int(x1), int(y1)), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (int(w1 + int(x2)), int(y2)), 3, (0, 255, 0), -1)
        cv2.line(canvas, (int(x1), int(y1)), (w1 + int(x2), int(y2)), (0, 255, 0), 1)
    return canvas


import numpy as np

vis = draw_simple_matches(img1, img2, pts1, pts2)
out = os.path.join(here, "..", "results", "demo_matches.png")
cv2.imwrite(out, vis)
print("Saved visualization to", out)
