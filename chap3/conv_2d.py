import argparse
import math

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

from utils import display_multi_img


def conv_2d(img, kernel, average=False):
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
            if average:
                out_img[i - pad, j - pad] /= kH * kW

    return out_img


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    iW = data.shape[1]
    iH = data.shape[0]
    data_final = []
    data_final = np.zeros((iH, iW))
    for i in range(iH):

        for j in range(iW):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > iH - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > iW - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    # np.linspace: Return evenly spaced numbers over a specified interval
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return conv_2d(image, kernel, average=True)


if __name__ == "__main__":
    # img = cv2.imread("./img/chest-x-ray.png", 0)
    img = cv2.imread("./img/lena_noise.png", 0)
    img_lena_noie = cv2.imread("./img/lena_with_noise.png", 0)
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
    # SOBEL_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # LAPLACIAN_2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # img_smooth = conv_2d(img, KERNEL_1)
    # img_sharpen1 = conv_2d(img, KERNEL_SHARPENING1)
    # display_multi_img([img, img_smooth, img_sharpen1])

    # img_sobelx = conv_2d(img, SOBEL_Y)
    # img_sobely = conv_2d(img, np.flip(SOBEL_Y.T, axis=0))
    # gradient_magnitude = np.sqrt(np.square(img_sobelx) + np.square(img_sobely))
    # img_laplacian = conv_2d(img, LAPLACIAN)
    # img_laplacian_2 = conv_2d(img, LAPLACIAN_2)
    # display_multi_img([img_sobelx, img_sobely, gradient_magnitude])
    # display_multi_img([img_laplacian, img_laplacian_2])

    # lena_noie_filtered = median_filter(img_lena_noie, 5)
    # display_multi_img([img_lena_noie, lena_noie_filtered])

    gaussian_blur_lena_img = gaussian_blur(img, 5)
    display_multi_img([img, gaussian_blur_lena_img])
