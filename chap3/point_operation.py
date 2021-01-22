import numpy as np
import cv2
import matplotlib.pyplot as plt

L = 256


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
            b_img[i, j] = c * np.log(img[i, j] + 1)

    return b_img


def plot_2_img(img1, img2, title=None):
    fig = plt.figure(num=title)
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img1, cmap="gray")
    ax.set_title("Before")
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img2, cmap="gray")
    ax.set_title("After")


if __name__ == "__main__":
    breast_img = cv2.imread("./img/Fig0304(a)(breast_digital_Xray).tif", 0)
    plot_2_img(breast_img, negative_ops(breast_img), "negative_ops")

    chest_img = cv2.imread("./img/chest-x-ray.png", 0)
    plot_2_img(chest_img, log_ops(chest_img), "chest_ops")
