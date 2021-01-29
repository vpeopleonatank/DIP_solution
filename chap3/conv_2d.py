from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
from utils import display_multi_img


def conv_2d(img, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    iH, iW, kH, kW = img.shape[0], img.shape[1], kernel.shape[0], kernel.shape[1]
    pad = (kW - 1) // 2
    out_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    padded_img = np.zeros(
        (img.shape[0] + pad * 2, img.shape[1] + pad * 2), dtype=np.uint8
    )
    padded_img[pad : iH + pad, pad : iW + pad] = img
    for i in range(pad, iH):
        for j in range(pad, iW):
            # get receptive field
            rc = padded_img[i - pad : i + pad + 1, j - pad : j + pad + 1]
            g = (rc * kernel).sum()
            out_img[i - pad, j - pad] = round(g)

    return out_img


if __name__ == "__main__":
    # img = cv2.imread("./img/chest-x-ray.png", 0)
    img = cv2.imread("./img/lena_noise.png", 0)
    KERNEL_1 = np.array(
        [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
    )
    KERNEL_SHARPENING1 = np.array(
        [
            [-1 / 9, -1 / 9, -1 / 9],
            [-1 / 9, 8 / 9, -1 / 9],
            [-1 / 9, -1 / 9, -1 / 9],
        ]
    )
    img_smooth = conv_2d(img, KERNEL_1)
    img_sharpen1 = conv_2d(img, KERNEL_SHARPENING1)
    display_multi_img([img, img_smooth, img_sharpen1])
