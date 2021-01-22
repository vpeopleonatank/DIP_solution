import numpy as np
import cv2


BIN = 8
L = 256

img = cv2.imread('./lena.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('', img)
cv2.waitKey(0)
