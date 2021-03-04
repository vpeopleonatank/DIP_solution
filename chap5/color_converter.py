import cv2
import numpy as np

class CVT_CODE:
    RGB2YCBCR = 1
    RGB2HSV = 2

def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v

def rgb_to_hsv_vectorized(img): # img with BGR format
    maxc = img.max(-1)
    minc = img.min(-1)

    out = np.zeros(img.shape)
    out[:,:,2] = maxc
    out[:,:,1] = (maxc-minc) / maxc

    divs = (maxc[...,None] - img)/ ((maxc-minc)[...,None])
    cond1 = divs[...,0] - divs[...,1]
    cond2 = 2.0 + divs[...,2] - divs[...,0]
    h = 4.0 + divs[...,1] - divs[...,2]
    h[img[...,2]==maxc] = cond1[img[...,2]==maxc]
    h[img[...,1]==maxc] = cond2[img[...,1]==maxc]
    out[:,:,0] = (h/6.0) % 1.0

    out[minc == maxc,:2] = 0
    return out

def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v

def rgb_to_hsv_loopy(img): 
    out_loopy = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):   
            out_loopy[i,j] = rgb_to_hsv(img[i,j,2],img[i,j,1],img[i,j,0])
    return out_loopy

def cvt_color(img, cvt_code):
    out = np.zeros_like(img)
    if cvt_code == CVT_CODE.RGB2YCBCR:
        b = np.array([16, 128, 128]).reshape(3)
        co_mat  = 1 / 255 * np.array([[65.481, 128.553, 24.966],
                            [-37.797, -74.203, 112.000],
                            [112.000, -93.786, -18.214]])

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j, :] = b + np.matmul(co_mat, img[i, j, :].reshape(3, 1)).reshape(3)

        cv2.imshow('out1', out[:, :, 0])
        cv2.imshow('out2', out[:, :, 1])
        cv2.imshow('out3', out[:, :, 2])

        # h x w x 3 * 3 x 3
        # print(img[1,1,:].shape)
        # res = img * co_mat[None, :, :]
        # res = np.einsum('ijk,kk->ijk',img, co_mat)
        # res = res + b
        # cv2.imshow('res', res)
    elif cvt_code == CVT_CODE.RGB2HSV:
        # source
        # https://stackoverflow.com/questions/38055065/efficient-way-to-convert-image-stored-as-numpy-array-into-hsv
        im = img / 255

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ma = np.max(im[i, j, :])
                mi = np.min(im[i, j, :])
                delta = ma - mi
                V = ma
                out[i, j, 2] = V 
                R = im[i, j, 0]
                G = im[i, j, 1]
                B = im[i, j, 2]
                if ma == mi:
                    H = 0
                elif ma == R:
                    H = 60 * (G - B) * delta
                elif ma == G:
                    H = 60 * (2 + (B - R) / delta)
                else:
                    H = 60 * (4 + (R - G) / delta)

                if H < 0:
                    H = H + 360
                out[i, j, 0] = H

                if ma != 0:
                    S = delta / ma
                else:
                    S = 0
                out[i, j, 1] = S

                # out[i, j, 0], out[i, j, 1], out[i, j, 2] = rgb_to_hsv(R, G, B)



        # print(out)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # print(np.allclose(rgb_to_hsv_vectorized(img), rgb_to_hsv_loopy(img)))
        # back = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        # cv2.imshow('back', back)
        cv2.imshow('hsv', hsv)

        cv2.imshow('out', out)
    cv2.imshow('img', img)



def main():
    img = cv2.imread('./images/peppers.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cvt_color(img, CVT_CODE.RGB2HSV)


if __name__ == '__main__':
    main()

