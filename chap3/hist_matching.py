import numpy as np
import cv2
import matplotlib.pyplot as plt
from img_2_hist import get_hist
from hist_equalize import get_cumulative_hist, get_pdf
from point_operation import plot_2_hist, plot_2_img

L = 256
M = 256


def get_matching_hist_img(org, specified):
    org_h = get_hist(org, M=M)
    specified_h = get_hist(specified, M=M)
    tfs = []
    for i in range(org_h):
        pass


if __name__ == "__main__":
    org = cv2.imread("./img/opencv.png", 0)
    specified = cv2.imread("./img/specified.png", 0)
    org_h = get_hist(org, M=M)
    specified_h = get_hist(specified, M=M)

    org_p = get_pdf(org_h, org.shape[0] * org.shape[1])
    specified_p = get_pdf(specified_h, specified.shape[0] * specified.shape[1])
    org_c = get_cumulative_hist(org_p)
    specified_c = get_cumulative_hist(specified_p) * (L - 1)
    org_c = [int(o * (L - 1)) for o in org_c]

    # plot_2_hist(org, specified)
    # cv2.imshow('', specified)
    # cv2.waitKey(0)
