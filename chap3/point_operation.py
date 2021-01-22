import numpy as np
import cv2
import matplotlib.pyplot as plt

L = 256

def negative_ops(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = L - img[i, j]

    return img

def plot_2_img(img1, img2):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img1, cmap="gray")
    ax.set_title('Before')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img2, cmap="gray")
    ax.set_title('After')

if __name__ == "__main__":
    breast_img = cv2.imread('./img/Fig0304(a)(breast_digital_Xray).tif', 0)
# cv2.imshow('org img', breast_img)
# cv2.waitKey()
    plot_2_img(breast_img, breast_img)
