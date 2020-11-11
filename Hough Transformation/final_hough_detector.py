import cv2
import random
import numpy as np
from File import Dict_Graph
from helper_functions import Helper_Fuctions
from poly_point_isect import isect_segments as isect_finder

from PIL import Image
# import pytesseract
import math
import os

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

HOUGH_BIG_LINE_PARAMETERS = {
    'kernel_size': 5,  # Default 5
    'low_threshold': 50,   # Default 50
    'high_threshold': 150, # Default 150
    'rho': 0.5,  # distance resolution in pixels of the Hough grid                     Default 1
    'theta': np.pi / 2000,  # angular resolution in radians of the Hough grid         Default np.pi / 180
    'threshold': 20,  # minimum number of votes (intersections in Hough grid cell)   Default 15
    'min_line_length': 50,  # minimum number of pixels making up a line              Default 50
    'max_line_gap': 5,  # maximum gap in pixels between connectable line segments   Default 20
}

HOUGH_SMALL_LINE_PARAMETERS = {
    'kernel_size': 5,  # Default 5
    'low_threshold': 50,   # Default 50
    'high_threshold': 150, # Default 150
    'rho': 0.5,  # distance resolution in pixels of the Hough grid                     Default 1
    'theta': np.pi / 2000,  # angular resolution in radians of the Hough grid         Default np.pi / 180
    'threshold': 15,  # minimum number of votes (intersections in Hough grid cell)   Default 15
    'min_line_length': 25,  # minimum number of pixels making up a line              Default 50
    'max_line_gap': 15,  # maximum gap in pixels between connectable line segments   Default 20
}

BIG_LINE_DELETE_INTENSITY_THRESH = 0.85
SMALL_LINE_DELETE_INTENSITY_THRESH = 0.95

def show_im(image, name='empty'):
    # Show cv2 image
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def houghLineDetector(gray, parameters):
    # Get gray cv2 image with hough parameters. Return detected lines
    blur_gray = cv2.GaussianBlur(gray, (parameters['kernel_size'], parameters['kernel_size']),0)
    edges = cv2.Canny(blur_gray, parameters['low_threshold'], parameters['high_threshold'])

    lines = cv2.HoughLinesP(
        edges, parameters['rho'], parameters['theta'], parameters['threshold'], 
        np.array([]), parameters['min_line_length'], parameters['max_line_gap']
    )
    return np.squeeze(lines)

def line_delete(gray, lines, threshold=0.3):
    # Get gray cv2 image with lines. Return cleaned image
    hp = Helper_Fuctions()
    dg = Dict_Graph()

    lines_dict = hp.lines_to_dict(lines)
    for idx, line in enumerate(lines_dict.values()):
        pixels, orient = dg.get_line_bounding_box(line)
        if dg.get_proj_percent(pixels, orient, gray) > threshold:
            for x,y in pixels:
                gray[y,x] = 255

    return gray
'''
def tesseract_char_delete(gray):
    # Get gray cv2 image. Return cleaned image

    filename = "/tmp/{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    bboxes_str = pytesseract.image_to_boxes(Image.open(filename))
    os.remove(filename)

    for bbox_str in bboxes_str.split('\n'):
        if len(bbox_str) > 4:
            char, start_x, start_y, end_x, end_y, thickness = bbox_str.split(' ')
        else:
            continue
        if char.isalnum() or char == ',':
            y_offset = gray.shape[1]
            bbox_start = (int(start_x), y_offset-int(start_y))
            bbox_end = (int(end_x), y_offset-int(end_y))
            cv2.rectangle(gray, bbox_start, bbox_end, (255,255,255), -1)
    
    return gray
'''
def draw_lines(image, lines, threshold=0.3):
    # Get cv2 image. Draw lines on image and return the image.

    hp = Helper_Fuctions()
    dg = Dict_Graph()
    lines_dict = hf.lines_to_dict(lines)

    for idx, line in enumerate(lines_dict.values()):
        pixels, orient = dg.get_line_bounding_box(line)
        if dg.get_proj_percent(pixels, orient, image) > threshold:
            x1, y1, x2, y2, orient = line
            r,g,b = [random.randint(25, 255), random.randint(0, 150), random.randint(0, 255)]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (r,g,b), 1)

    return image

images_new = [
    "0030_f_0_0.png", "0119_t_0_0.png", "0336_f_0_0.png", "0757_t_0_0.png", "0933_t_0_0.png",
    "0031_f_0_0.png", "0265_f_0_0.png", "0336_s_0_0.png", "0769_f_0_0.png", "0935_f_0_0.png",
    "0053_f_0_0.png", "0265_s_0_0.png", "0727_f_0_0.png", "0769_s_0_0.png", "0935_t_0_0.png",
    "0053_t_0_0.png", "0265_t_0_0.png", "0727_t_0_0.png", "0769_t_0_0.png", "0971_f_0_0.png",
    "0115_f_0_0.png", "0303_f_0_0.png", "0757_f_0_0.png", "0933_f_0_0.png", "0971_s_0_0.png",
    "0119_f_0_0.png", "0303_t_0_0.png", "0757_s_0_0.png", "0933_s_0_0.png", "0971_t_0_0.png",
]

for image in images_new:
    INPUT_FILE = image
    main_image = cv2.imread(INPUT_FILE)
    line_image = np.copy(main_image) * 0

    ## Hough 1 ##########################################################################################
    ## Detect big lines here ############################################################################

    gray = cv2.cvtColor(main_image.copy(), cv2.COLOR_BGR2GRAY)
    big_lines = houghLineDetector(gray.copy(), HOUGH_BIG_LINE_PARAMETERS)

    ## Line Delete ######################################################################################
    ## Delete big lines from image so we reduce complexity ##############################################

    big_lines_cleaned_gray = line_delete(gray.copy(), big_lines.copy(), BIG_LINE_DELETE_INTENSITY_THRESH)

    ## Merge Lines ######################################################################################
    ## Merge double lines ###############################################################################

    hf = Helper_Fuctions()
    merged_big_lines = hf.merge_lines(big_lines.copy())

    ## Draw Lines #######################################################################################
    ## Draw big lines on the image ######################################################################

    line_image = draw_lines(line_image.copy(), big_lines.copy(), threshold=BIG_LINE_DELETE_INTENSITY_THRESH)
    blended = cv2.addWeighted(main_image.copy(), 0.4, line_image.copy(), 0.9, 0)
    #show_im(blended)

    ## OCR ##############################################################################################
    ## To get rid of some numbers or characters #########################################################

    # char_cleaned_gray = tesseract_char_delete(big_lines_cleaned_gray.copy())

    ## Hough 2 ##########################################################################################
    ## Detect the remaining small lines #################################################################

    small_lines = houghLineDetector(big_lines_cleaned_gray.copy(), HOUGH_SMALL_LINE_PARAMETERS)

    # If there is no small line
    if len(small_lines.shape) < 1:
        blended = cv2.addWeighted(main_image, 0.4, line_image, 0.9, 0)
        cv2.imwrite('blended_'+INPUT_FILE, blended)
        continue

    ## Merge Lines ######################################################################################
    ## Merge double lines ###############################################################################

    small_lines = np.expand_dims(small_lines.copy(), axis=0) if len(small_lines.shape) != 2 else small_lines.copy()
    merged_small_lines = hf.merge_lines(small_lines.copy())

    ## Draw Lines #######################################################################################
    ## Draw small lines on the image ####################################################################

    line_image = draw_lines(line_image.copy(), small_lines.copy(), threshold=SMALL_LINE_DELETE_INTENSITY_THRESH)

    ## Write results ####################################################################################
    ## Write results to the disk ########################################################################

    blended = cv2.addWeighted(main_image.copy(), 0.4, line_image.copy(), 0.9, 0)
    cv2.imwrite('blended_'+INPUT_FILE, blended)