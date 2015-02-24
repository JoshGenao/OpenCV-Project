__author__ = 'joshgenao'
import cv2
import numpy as np
from matplotlib import pyplot as plt

def FlannMatcher(queryImage, image):
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(queryImage,0)         # queryImage
    img2 = cv2.imread(image,0)              # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    # Processes the matches using FlannBasedMatcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # A good match will show a lot of matches
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    print len(good)
    pass

if __name__ == "__main__":
    FlannMatcher('images/Book.jpg', 'images/BookShift2.jpg')