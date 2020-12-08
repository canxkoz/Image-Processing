import cv2
import random
import numpy as np

from pprint import pprint

import matplotlib.pyplot as plt
from scipy.spatial import distance as dist


def find_island_cmass(island):
    # Get island and return cmass x,y
    M = cv2.moments(island)
    c_mass_x = int(M['m10']/M['m00'])
    c_mass_y = int(M['m01']/M['m00'])

    return c_mass_x, c_mass_y

def find_island_area(island):
    # Get island and return area
    M = cv2.moments(island)
    area = M['m00']
    
    return area

def find_island_coef_var(island, c_mass_x, c_mass_y):
    # Get island and return coef_var

    distances = []
    for black_point in island:
        # Get black point x and y
        point_X = black_point[0][0]
        point_Y = black_point[0][1]
        # Find distances between center of the bounding box and black point
        distance = dist.euclidean((c_mass_x, c_mass_y), (point_X, point_Y))

        distances.append(distance)
    
    distances = np.array(distances)
    # Find coef_var of the distances
    std = distances.std()
    mean = distances.mean()
    # Multiple with 10 to get better results
    return std/mean*10

def find_coef_var_outer_by_inner(inner_island, outer_island_bbox):
    # Get inner islands and outer islands bbox
    # Find https://en.wikipedia.org/wiki/Coefficient_of_variation for the outer islands by its inner islands
    # Return Coefficient_of_variation in dictionary format
    # {'island0': float, 'island1': float, ...}

    results = {}
    # Iterate on every outer island
    for i in range(len(outer_island_bbox)):
        bbox = outer_island_bbox[i]
        islands_in_bbox = []
        for island in inner_island:
            # Check if island in bbox
            # Get random point from inner island and check if in the outer island bbox
            x,y = island[random.randint(0, len(island)-1)][0]
            if cv2.pointPolygonTest(bbox, (x, y), False) > 0:
                islands_in_bbox.append(island)
                break

        # Compute the center of the outer island bounding box
        cXi = np.average(bbox[:, 0])
        cYi = np.average(bbox[:, 1])

        # Concat inner islands
        if len(islands_in_bbox) > 1:
            big_island = np.concatenate((island for island in islands_in_bbox), axis=0)
        else:
            big_island = islands_in_bbox[0]

        # Get Coefficient_of_variation
        coef_var = find_island_coef_var(big_island, cXi, cYi)

        # Put coef_var for every outer island
        results['island'+str(i)] = coef_var

    return results

def results_dictionary(outer_islands, distances, coef_vars):
    # Takes distances and coef_vars in dictionary format
    # Calculates center of mass and area
    # Arange and return all results in dictionary format
    # results = {'island0': {'c_mass':[x,y], 'area': float, 'distance':{'island1': float, 'island2': float, ...}, 'coef_var': float}, 'island1': ...}

    results = {}
    for i in range(len(outer_islands)):
        island = outer_islands[i]

        # Calculate c_mass and area
        c_mass_x, c_mass_y = find_island_cmass(island)
        area = find_island_area(island)

        # Get coef_var dict
        #coef_var = coef_vars['island'+str(i)]
        coef_var = find_island_coef_var(island, c_mass_x, c_mass_y)
        # Get distance dict
        distance = distances['island'+str(i)]

        results['island'+str(i)] = {'c_mass': [c_mass_x, c_mass_y], 'area': area, 'distance': distance, 'coef_var': coef_var}

    return results

def find_contour_bbox(contour):
    # Find and return bounding box of the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="int")

    return box

def find_contours(image, type='RETR_EXTERNAL'):
    # Find contours on the image
    # RETR_EXTERNAL = gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
    # RETR_LIST = gives all the contours and doesn't even bother calculating the hierarchy -- good if you only want the contours and don't care whether one is nested inside another.
    
    # Grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert
    gray = 255 - gray 

    # Find bounding box contours of polygons
    if type == 'RETR_LIST':
        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    elif type == 'RETR_EXTERNAL':
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def draw_contours(image, contours, fill=True):
    # Draw contours on given image and return image
    for poly in contours:
        # Fill every polygon w/ random color
        r,g,b = [random.randint(0, 255), random.randint(0, 230), random.randint(0, 255)]
        if fill:
            cv2.fillPoly(image, pts=[poly], color=(r,g,b))
        else:
            cv2.drawContours(image, [poly], -1, (r,g,b), 1)
            
    return image

def midpoint(ptA, ptB):
    # Find mid point of given two point
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def find_distance_between_islands(image, islands):
    # Find and show distances between islands
    # Return distances in dictionary format
    # {'island0': {'island1': float, 'island2': float, ...}, 'island1': {'island0': float, 'island2': float, ...}, ...}

    all_distances = {}
    for i in range(len(islands)):
        # compute the rotated bounding box of the island
        boxi = cv2.minAreaRect(islands[i])
        boxi = cv2.boxPoints(boxi)
        boxi = np.array(boxi, dtype="int")

        # compute the center of the bounding box
        cXi = np.average(boxi[:, 0])
        cYi = np.average(boxi[:, 1])

        distances = {}
        for j in range(len(islands)):

            if i == j:
                # We wont need distance between islandX and islandX
                continue

            # compute the rotated bounding box of the island
            boxj = cv2.minAreaRect(islands[j])
            boxj = cv2.boxPoints(boxj)
            boxj = np.array(boxj, dtype="int")

            # compute the center of the bounding box
            cXj = np.average(boxj[:, 0])
            cYj = np.average(boxj[:, 1])

            distance = dist.euclidean((cXi, cYi), (cXj, cYj))

            distances['island'+str(j)] = distance

        all_distances['island'+str(i)] = distances

    return all_distances

def dilation(image, filter, iteration):
    # Apply dilation
    kernel = np.ones((filter, filter), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=iteration)

    return dilation

def find_connected_islands(image, threshold=150, apply_dilation=True, dilation_filter=3, dilation_iteration=2):
    # Find connected components
    # Apply some noise reduction invert and dilation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Delete pixels if whiter then threshold
    ret, thresh = cv2.threshold(gray, threshold, 255, 0)

    # Invert image 
    thresh = 255-thresh
    
    if apply_dilation:
        # Apply dilation to close gap between circle parts
        thresh = dilation(thresh, dilation_filter, dilation_iteration)

    # Find connected components
    _, labels = cv2.connectedComponents(thresh)
    return labels


def filter_detected_islands(labels, threshold=8):
    # Filter and remove connected components which are smaller then threshold
    num = labels.max()
    for i in range(1, num+1):
        pts = np.where(labels == i)
        range_x_i = np.max(pts[0])-np.min(pts[0])
        range_y_i = np.max(pts[1])-np.min(pts[1])
        if range_x_i < threshold or range_y_i < threshold:  # Hyperparameter  default 13
            labels[pts] = 0
    return labels


def hsv_label2image(labels):
    # Recrate RGB image from HSV labels
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 255

    return labeled_img

if __name__ == "__main__":
    image_names = [
        "0030_f_0_0.png", "0119_t_0_0.png", "0336_f_0_0.png", "0757_t_0_0.png", "0933_t_0_0.png",
        "0031_f_0_0.png", "0265_f_0_0.png", "0336_s_0_0.png", "0769_f_0_0.png", "0935_f_0_0.png",
        "0053_f_0_0.png", "0265_s_0_0.png", "0727_f_0_0.png", "0769_s_0_0.png", "0935_t_0_0.png",
        "0053_t_0_0.png", "0265_t_0_0.png", "0727_t_0_0.png", "0769_t_0_0.png", "0971_f_0_0.png",
        "0115_f_0_0.png", "0303_f_0_0.png", "0757_f_0_0.png", "0933_f_0_0.png", "0971_s_0_0.png",
        "0119_f_0_0.png", "0303_t_0_0.png", "0757_s_0_0.png", "0933_s_0_0.png", "0971_t_0_0.png",
    ]

    for image_name in image_names:
        INPUT_FILE = './big_lines_deleted/big_deleted_' + image_name
        image = cv2.imread(INPUT_FILE)

        # Keep original image
        image_original = image.copy()

        # White empty image
        white_image = image.copy()
        white_image.fill(255)

        # Find connected components
        labels = find_connected_islands(image, apply_dilation=True, dilation_filter=3, dilation_iteration=2)
        
        # Filter and remove too small components
        labels = filter_detected_islands(labels)

        # ReCreate BGR image w/ using HSV labels
        labeled_img = hsv_label2image(labels)

        # Get contours of the polygons
        # contours_dilated = Outer islands, only external contours, use as bound box 
        contours_dilated = find_contours(labeled_img, type='RETR_EXTERNAL')
        # contours_undilated = Inner islands, not dilated, external and internal contours
        contours_undilated = find_contours(image_original, type='RETR_LIST')

        # Get bbox of the contours_dilated
        # contours_dilated_bbox = outer islands bbox
        contours_dilated_bbox = [find_contour_bbox(contour) for contour in contours_dilated]

        # Draw polygons with random colors
        # image_dilated = Outer islands
        image_dilated = draw_contours(white_image.copy(), contours_dilated, fill=False)
        # image_undilated = Inner islands
        image_undilated = draw_contours(white_image.copy(), contours_undilated, fill=False)

        # Find distances between outer islands
        outer_islands_distances = find_distance_between_islands(image, contours_dilated)

        # Find https://en.wikipedia.org/wiki/Coefficient_of_variation for the outer islands by its inner islands
        outer_islands_coef_var = find_coef_var_outer_by_inner(contours_undilated, contours_dilated_bbox)

        # Get findings in dictionary format
        # Results contains distances, center of mass, areas and coef_vars
        # center of mass and area calculated in results()
        results = results_dictionary(contours_dilated, outer_islands_distances, outer_islands_coef_var)

        pprint(results)