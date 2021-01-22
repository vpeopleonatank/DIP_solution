import numpy as np
import cv2
import matplotlib.pyplot as plt


M = 8  # nums of bin
L = 256

DELTA = (L - 1) / M


def get_pixel_bin(v):
    for i in range(M):
        lower_d = DELTA * i
        if i != M:
            upper_d = DELTA * (i + 1)
        else:
            upper_d = DELTA * (i + 1) + 1
        if lower_d <= v and v < upper_d:
            return i

    return -1


def get_hist(img):
    h = [0] * M
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if get_pixel_bin(img[i, j]) != -1:
                h[get_pixel_bin(img[i, j])] += 1

    return h


if __name__ == "__main__":

    img = cv2.imread("./lena.png", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("", img)
# cv2.waitKey(0)

    b = [j for j in range(1, 9)]
    hist = get_hist(img)
    hist = np.ravel(img)
    print(hist.shape)
    fig = plt.figure()
    plt.hist(hist, bins=8)
    plt.xlabel('bin')
    plt.ylabel('number of pixels')
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
