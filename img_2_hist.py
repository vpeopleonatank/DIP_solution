import numpy as np
import cv2


M = 8  # nums of bin
L = 256

DELTA = (L - 1) / M


def get_pixel_bin(v):
    for i in range(M):
        lower_d = DELTA * i
        if i != M:
            upper_d = DELTA * (i + 1)
        else:
            upper_d = DELTA * (i + 1) + 1
        if lower_d <= v and v < upper_d:
            return i

    return -1

def get_hist(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            get_pixel_bin(img[i, j])

img = cv2.imread("./lena.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("", img)
cv2.waitKey(0)
