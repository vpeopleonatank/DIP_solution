import numpy as np
import cv2
import matplotlib.pyplot as plt


M = 256  # nums of bin
L = 256

DELTA = (L - 1) / M


def get_pixel_bin(v, BINS=M):
    for i in range(BINS):
        lower_d = DELTA * i
        if i != BINS:
            upper_d = DELTA * (i + 1)
        else:
            upper_d = DELTA * (i + 1) + 1
        if lower_d <= v and v < upper_d:
            return i

    return -1


def get_hist(img, M=M):
    h = [0] * M
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bin_th = get_pixel_bin(img[i, j], BINS=M)
            if bin_th != -1:
                h[bin_th] += 1

    return h


def count_diff(a1, a2):
    c = 0
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            c += 1

    return c


def count_pixels_inrange(img, lo, up):
    return np.sum((img > lo) & (img <= up))  # 31 63.75


if __name__ == "__main__":

    img = cv2.imread("./lena.png", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("", img)
    # cv2.waitKey(0)

    b = [j for j in range(1, 9)]
    own_hist = get_hist(img)
    cv2_hist = cv2.calcHist([img], [0], None, [M], [0, M])
    # fig = plt.figure()
    # n, bins, _ = plt.hist(np.ravel(img), bins=M)  # not correct
    # plt.hist(cv2_hist, BINS, [0, BINS])
    # plt.plot(cv2_hist, color="r")
    plt.plot(own_hist, color="b")
    # plt.xlabel("bin")
    # plt.ylabel("number of pixels")
    plt.show()


# %%
# import matplotlib.pyplot as plt
# import numpy as np
#
# np.random.seed(19680801)
# data = np.random.randn(2, 100)
#
# fig, axs = plt.subplots(2, 2, figsize=(5, 5))
# axs[0, 0].hist(data[0])
# axs[1, 0].scatter(data[0], data[1])
# axs[0, 1].plot(data[0], data[1])
# axs[1, 1].hist2d(data[0], data[1])
#
# plt.show()
