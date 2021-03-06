import numpy as np
import cv2
import matplotlib.pyplot as plt

L = 256
M = 256


def negative_ops(img):
    b_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_img[i, j] = L - img[i, j] - 1

    return b_img


def identity(img):
    b_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_img[i, j] = img[i, j]

    return b_img


def log_ops(img):
    b_img = np.zeros((img.shape[0], img.shape[1]))
    c = (L - 1) / np.log(L)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_img[i, j] = c * np.log(img[i, j] + 1)

    return b_img


def powerlaw_ops(img, gamma=0.04):
    b_img = np.zeros((img.shape[0], img.shape[1]))
    c = np.power((L - 1), 1 - gamma)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_img[i, j] = c * np.power(img[i, j], gamma)

    return b_img


def scaling_ops(img, r1, s1, r2, s2):
    b_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_img[i, j] = s2 * ((img[i, j] - r1) / (r2 - r1))

    return b_img


def piece_wise_linear_ops(img, s1, s2, r1, r2):
    b_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    k1 = s1 / r1
    k2 = (s2 - s1) / (r2 - r1)
    k3 = (L - 1 - s2) / (L - 1 - r2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j]
            if x >= 0 and x < r1:
                b_img[i, j] = k1 * x
            elif x >= r1 and x < r2:
                b_img[i, j] = (x - r1) * k2 + s1
            elif x >= r2 and x <= (L - 1):
                b_img[i, j] = (x - r2) * k3 + s2

    return b_img


def thresholding(img, tau, value):
    b_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j]
            if x >= tau:
                b_img[i, j] = value
            else:
                b_img[i, j] = 0

    return b_img


def plot_2_img(img1, img2, title=None):
    fig = plt.figure(num=title)
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img1, cmap="gray")
    ax.set_title("Before")
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img2, cmap="gray")
    ax.set_title("After")


def plot_2_hist(img1, img2, title=None):
    fig = plt.figure(num=title)
    ax = fig.add_subplot(1, 2, 1)
    histr1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    imgplot = plt.plot(histr1, color="r")
    plt.xlim([0, 256])
    # plt.ylim([0, img1.shape[0]])
    ax.set_title("Before")
    ax = fig.add_subplot(1, 2, 2)
    histr2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    imgplot = plt.plot(histr2, color="b")
    plt.xlim([0, 256])
    # plt.ylim([0, img1.shape[0]])
    ax.set_title("After")


if __name__ == "__main__":
    breast_img = cv2.imread("./img/Fig0304(a)(breast_digital_Xray).tif", 0)
    plot_2_img(breast_img, negative_ops(breast_img), "negative_ops")
    #
    # chest_img = cv2.imread("./img/chest-x-ray.png", 0)
    # plot_2_img(chest_img, log_ops(chest_img), "chest_ops")

    # ariel_img = cv2.imread("./img/Fig0309(a)(washed_out_aerial_image).tif", 0)
    # plot_2_img(ariel_img, powerlaw_ops(ariel_img, gamma=2.5), "power-law gamma 2.5 ops")
    # plot_2_img(ariel_img, powerlaw_ops(ariel_img, gamma=5.), "power-law gamma 5 ops")

    # da_img = cv2.imread("./img/Fig0320(4)(bottom_left).tif", 0)
    # mi = np.min(da_img)
    # ma = np.max(da_img)
    # stretched_da_img = scaling_ops(da_img, r1=mi, s1=0, r2=ma, s2=L - 1)
    # plot_2_hist(da_img, stretched_da_img)
    # plot_2_img(da_img, stretched_da_img)
    # linear_da_img = piece_wise_linear_ops(da_img, s1=0, s2=L-1, r1=da_img.min(), r2=da_img.max())
    # plot_2_hist(da_img, linear_da_img)
    # plot_2_img(da_img, linear_da_img)

    # threshold_da_img = thresholding(da_img, tau=38, value=L-1)
    # plot_2_hist(da_img, threshold_da_img)
    # plot_2_img(da_img, threshold_da_img)
