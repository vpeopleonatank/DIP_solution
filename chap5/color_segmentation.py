import cv2
import numpy as np


def main():
    img = cv2.imread('./images/Fig0642(a)(jupiter_moon_original).tif', cv2.IMREAD_UNCHANGED)
    cv2.imshow('img', img)
    # 250, 60 -> 324, 103
    roi = img[250: 324, 60: 103, :]
    mean_vector = np.array([np.mean(roi[:,:,i]) for i in range(3)])
    # cov_mat = np.array([np.cov(roi[:, :, i] for i in range(3))])
    print(np.cov(roi[:, :, 0]).shape)


if __name__ == "__main__":
    main()
