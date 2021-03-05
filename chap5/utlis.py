import cv2
import numpy as np



def main():
    # H x W x 3 * H x W x 3
    img = np.arange(192).reshape(8, 8, 3)
    meanI = np.array([np.mean(img[:, :, i]) for i in range(3)])
    img = img - meanI
    lenI = img.shape[2]
    print(img.shape)
    print(img.T.dot(img).shape)
    cov_mat = np.tensordot(img.T, img, 2) / lenI

    print(cov_mat.shape)



if __name__ == '__main__':
    main()
