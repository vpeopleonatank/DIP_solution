import numpy as np
import cv2
import matplotlib.pyplot as plt
from img_2_hist import get_hist
from hist_equalize import get_cumulative_hist, get_pdf
from point_operation import plot_2_hist, plot_2_img
from utils import display_multi_img

L = 256
M = 256


def get_matching_hist_img(img, org_c, specified_c):
    tfs = []
    matched_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    mapping_table = []
    for i in org_c:
        for s in range(len(specified_c)):
            if i <= specified_c[s]:
                mapping_table.append(s)
                break

    import ipdb; ipdb.set_trace()
    for i in range(org.shape[0]):
        for j in range(org.shape[1]):
            matched_img[i, j] = mapping_table[img[i, j]]

    return matched_img


def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c  # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()


def hist_match(original, specified):

    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(
        original, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(specified, return_counts=True)

    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Round the values
    sour = np.around(s_quantiles * 255)
    temp = np.around(t_quantiles * 255)

    # Map the rounded values
    b = []
    for data in sour[:]:
        b.append(find_nearest_above(temp, data))
    b = np.array(b, dtype="uint8")

    return b[bin_idx].reshape(oldshape)


if __name__ == "__main__":
    org = cv2.imread("./img/opencv.png", 0)
    specified = cv2.imread("./img/specified.png", 0)
    org_h = get_hist(org, M=M)
    specified_h = get_hist(specified, M=M)

    org_p = get_pdf(org_h, org.shape[0] * org.shape[1])
    specified_p = get_pdf(specified_h, specified.shape[0] * specified.shape[1])
    # import ipdb; ipdb.set_trace()
    org_c = get_cumulative_hist(org_p)
    specified_c = get_cumulative_hist(specified_p)
    org_c = [round(o * (L - 1)) for o in org_c]
    specified_c = [round(o * (L - 1)) for o in specified_c]
    matched_img = get_matching_hist_img(org, org_c, specified_c)
    # plot_2_hist(org, specified)
    a = hist_match(org, specified)
    display_multi_img([org, specified, matched_img, a])
    # cv2.imshow("", matched_img)
    # cv2.waitKey(0)
