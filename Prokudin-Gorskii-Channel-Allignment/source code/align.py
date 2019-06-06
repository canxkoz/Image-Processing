import numpy as np
from scipy import misc
from skimage.transform import rescale
from skimage.io import imread, imsave, imshow
from os import mkdir
from os.path import dirname, exists
from sklearn.metrics import mean_squared_error


def norm_cc(im1, im2):
    return np.sum(im1 * im2) / np.sqrt(np.sum(im1 * im1) * np.sum(im2 * im2))


def ind_by_offset(offset, length):
    a_b = a_e = b_b = b_e = 0  # (matr_coord, begin/end)

    a_b = 0
    a_e = length - abs(offset)
    b_b = abs(offset)
    b_e = length

    if offset > 0:
        a_b, b_b = b_b, a_b
        a_e, b_e = b_e, a_e

    return (a_b, a_e), (b_b, b_e)


def image_offset(im1, im2, num_pix):
    offset_x = offset_y = -num_pix
    metric_max = 0

    for x in range(-num_pix, num_pix + 1):
        for y in range(-num_pix, num_pix + 1):
            im1_x, im2_x = ind_by_offset(x, im1.shape[0])
            im1_y, im2_y = ind_by_offset(y, im1.shape[1])

            metric = norm_cc(im1[im1_x[0]: im1_x[1], im1_y[0]: im1_y[1]],
                             im2[im2_x[0]: im2_x[1], im2_y[0]: im2_y[1]])

            if x == -num_pix and y == -num_pix or metric > metric_max:
                metric_max = metric
                offset_x = x
                offset_y = y

    return np.asarray([offset_x, offset_y])


def pyramid(im1, im2, level):
    if level == 0:
        return image_offset(im1, im2, 5)

    im1_res = rescale(im1, 0.5)
    im2_res = rescale(im2, 0.5)

    offset = 2 * pyramid(im1_res, im2_res, level - 1)

    im2 = np.roll(im2, offset[0], axis=0)
    im2 = np.roll(im2, offset[1], axis=1)

    offset += image_offset(im1, im2, 1)
    return offset


def align(bgr_image, g_coord=None):
    """Aligns an image from BGR channels given as one image
       
       Parameters 
       ----------
       bgr_image : NumPy 3d-array
           An image containing BGR channels placed in a column.

       g_coord : list
           x,y coordinates of a pooint in green channel for giving an offset
           relatively to it.
      
       Returns
       ----------
       al_im : NumPy 3d-array
           Aligned image
       
       b_coord : list
           x, y offset of blue channel  relativly to g_coord
       
       g_coord : list
           x, y offset of red channel relativly to g_coord
    """

    b_row = b_col = r_row = r_col = 0

    h_div3 = bgr_image.shape[0] // 3
    blue, green, red = bgr_image[0: h_div3, :], bgr_image[h_div3: 2 * h_div3, :], \
                       bgr_image[2 * h_div3: 3 * h_div3, :]

    channels = [blue, green, red]

    h, l = channels[0].shape[0], channels[0].shape[1]
    h_slice, l_slice = h * 5 // 100, l * 5 // 100
    for i in range(3):
        channels[i] = channels[i][h_slice: h - h_slice, l_slice: l - l_slice]

    if (channels[0].shape[0] <= 500):
        offset_bg = image_offset(channels[1], channels[0], 15)
        offset_rg = image_offset(channels[1], channels[2], 15)
    else:
        offset_bg = pyramid(channels[1], channels[0], 5)
        offset_rg = pyramid(channels[1], channels[2], 5)

    channels[0] = np.roll(channels[0], offset_bg[0], axis=0)
    channels[0] = np.roll(channels[0], offset_bg[1], axis=1)

    channels[2] = np.roll(channels[2], offset_rg[0], axis=0)
    channels[2] = np.roll(channels[2], offset_rg[1], axis=1)

    channels[0], channels[2] = channels[2], channels[0]
    al_im = misc.toimage(channels)
   
    if g_coord != None:
        b_row = g_coord[0] - offset_bg[0] - h_div3
        b_col = g_coord[1] - offset_bg[1]
        r_row = g_coord[0] - offset_rg[0] + h_div3
        r_col = g_coord[1] - offset_rg[1]

        b_coord = (b_row, b_col)
        r_coord = (r_row, r_col)
        return al_im, b_coord, r_coord
    return al_im

