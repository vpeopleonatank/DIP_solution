import numpy as np
import cv2
import matplotlib.pyplot as plt
from img_2_hist import get_hist
from point_operation import plot_2_hist, plot_2_img

L = 256
M = 256


def get_pdf(hist, N):
    return [h / N for h in hist]


def get_cumulative_hist(hist):
    c_n = [0] * len(hist)
    c_n[0] = hist[0]
    for i in range(1, len(hist)):
        c_n[i] = c_n[i - 1] + hist[i]

    return c_n


def hist_equalize(img):
    hist = get_hist(img, M=M)
    p_n = get_pdf(hist, img.shape[0] * img.shape[1])
    c_n = get_cumulative_hist(p_n)

    b_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_img[i, j] = round(((L - 1) * c_n[img[i, j]]))

    return b_img


if __name__ == "__main__":
    img = cv2.imread("./lena.png", cv2.IMREAD_GRAYSCALE)
    cv_eq_img = cv2.equalizeHist(img)
    equaled_img = hist_equalize(img)
    # plot_2_img(img, equaled_img)
    plot_2_hist(cv_eq_img, equaled_img)

    # plt.plot(own_hist, color="b")
    plt.show()
