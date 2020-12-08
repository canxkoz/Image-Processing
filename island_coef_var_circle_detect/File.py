import cv2
import random
import numpy as np

from poly_point_isect import isect_segments as isect_finder
from numpy.linalg import norm

import math
import queue
import statistics


class Dict_Graph:

    def __init__(self):
        pass

    def get_line_bounding_box(self, line_dict_value):
        '''
        Get the bounding box for a line.
        args:
            line_dict_value: value from dictionary containing line information {'line0': [x1, y1, x2, y2, 'Vrt']}
                            line_dict_value is a list containing [x1, y1, x2, y2, 'Information of line']
        return:
            bounding_box: a list contianing all pixels within the bounding box
                     [[x1,y1], [x2,y2], [x3, y3], [x4, y4]]
            orient: determine the direction to do black pixel summation. It could be horizontal: 'Hrz' or vertical: 'Vrt'.
        '''
        orient = ' '
        value = line_dict_value
        bounding_box = []
        box_range = 4
        x1 = value[0]
        y1 = value[1]
        x2 = value[2]
        y2 = value[3]
        min_x = int(min(x1, x2) - box_range)
        max_x = int(max(x1, x2) + box_range)
        min_y = int(min(y1, y2) - box_range)
        max_y = int(max(y1, y2) + box_range)

        if value[4] == 'Vrt':
            orient = 'Vrt'
        if value[4] == 'Hrz':
            orient = 'Hrz'

        if value[4] == 'Vrt' or value[4] == 'Hrz' or x1 == x2 or y1 == y2:
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    bounding_box.append([x, y])
        else:
            # find the line equation: y = kx + b
            k = (y2 - y1) / (x2 - x1)
            if abs(k) <= 1:
                orient = 'Hrz'
            else:
                orient = 'Vrt'
            b = y1 - k * x1
            # using bfs
            q = queue.Queue(0)
            visited = []
            start_x = int(min(x1,x2))
            start_y = int(min(y1,y2))
            end_x = int(max(x1,x2))
            end_y = int(max(y1,y2))
            q.put([int(x1), int(y1)])
            visited.append([int(x1), int(y1)])
            # possible_move = [[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1]]
            possible_move = [[1,0],[1,1],[-1,0],[-1,-1]]

            while q.empty() != True:
                cur = q.get()
                bounding_box.append([cur[0],cur[1]])
                for move in possible_move:
                    new_x = cur[0] + move[0]
                    new_y = cur[1] + move[1]
                    # new_count = cur[2]
                    if abs(k*new_x - new_y + b) / math.sqrt(k*k + 1) <= box_range and [new_x, new_y] not in visited \
                    and new_x <= end_x + box_range and new_x >= start_x - box_range  and new_y <= end_y + box_range \
                    and new_y >= start_y - box_range:
                        visited.append([new_x, new_y])
                        q.put([new_x, new_y])
        return bounding_box, orient


    def get_proj_percent(self, bounding_box, orient, img):
        '''
        Helper function to project black pixels towards axis and return the percentage of number of
        black pixels on x or y axis.
        args:
            bounding_box: list of points within the bounding box [[x1,y1], [x2,y2], [x3, y3], ......]
            orient: Direction to perform projection.
                    It could be horizontal: 'Hrz' or vertical: 'Vrt'
            img: cv2 image
        return:
            percentage: the percentage of number of black pixels along the projection line. must <= 1
        '''
        img_sum = {}
        ori_key = 0
        if orient == 'Hrz': # do summation to the x axis (horizontal axis).
            ori_key = 0
        else:
            ori_key = 1
        for point in bounding_box:
            img_sum[point[ori_key]] = 0
        for point in bounding_box:
            non_black = np.count_nonzero(img[point[1]][point[0]])
            if non_black == 0:
                img_sum[point[ori_key]] = 1

        total_black = 0
        for black in img_sum.values():
            total_black += black
        return total_black / len(img_sum)

