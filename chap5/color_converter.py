import cv2
import numpy as np

class CVT_CODE:
    RGB2YCBCR = 1

def cvt_color(img, cvt_code):
    if cvt_code == CVT_CODE.RGB2YCBCR:
        b = np.array([16, 128, 128]).reshape(1, 1, 3)
        co_mat  = 1 / 255 * np.array([[65.481, 128.553, 24.966],
                            [-37.797, -74.203, 112.000],
                            [112.000, -93.786, -18.214]])
        # h x w x 3 * 3 x 3
        # print(img[1,1,:].shape)
        # res = img * co_mat[None, :, :]
        res = np.einsum('ijk,kk->ijk',img, co_mat)
        res = res + b
        print(res.shape)

def main():
    img = cv2.imread('./images/peppers.png')
    cvt_color(img, CVT_CODE.RGB2YCBCR)


if __name__ == '__main__':
    main()

