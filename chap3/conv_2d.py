from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def conv_2d(img, kernel):
    iW, iH, kW, kH = img.shape[0], img.shape[1], kernel.shape[0], kernel.shape[1]
    pad = (kW - 1) // 2
    out_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.unit8)
    padded_img = np.zeros((img.shape[0] + pad, img.shape[1] + pad), dtype=np.unit8)
    for i in range(pad, iH):
        for j in range(pad, iW):
            # get receptive field
            rc = padded_img[i - pad : i + pad + 1, j - pad : j + pad + 1]
            g = (rc * kernel).sum()
            out_img[i]


if __name__ == "__main__":
    img = cv2.imread("./img/chest-x-ray.png", 0)
