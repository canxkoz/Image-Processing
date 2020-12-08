import numpy as np
from numpy import inf
import math
import cv2

class Helper_Fuctions():
    def __init__(self):
        pass

    def euclidean_dist(self, x1,y1,x2,y2):
        '''
        Calc euclidean distance between two points
        args:
            x1: int
            y1: int
            x2: int
            y2: int
        return:
            distance in pixel: float
        '''
        return math.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

    def calc_min_distance(self,line1_start,line1_end,line2_start,line2_end):
        '''
        Calc min distance between two line segment

        notes:
            if you work on 2D plane use 0 as z axis on all points
        args:
            line1_start: 1x3 np.array
            line1_end: 1x3 np.array
            line2_start: 1x3 np.array
            line2_end: 1x3 np.array
        return:
            distance in pixel: float

        '''

        # Calculate denomitator
        A = line1_end - line1_start
        B = line2_end - line2_start
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)
        
        _A = A / magA
        _B = B / magB
        
        cross = np.cross(_A, _B);
        denom = np.linalg.norm(cross)**2
        
        
        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
            d0 = np.dot(_A,(line2_start-line1_start))
            d1 = np.dot(_A,(line2_end-line1_start))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return np.linalg.norm(line1_start-line2_start)
                return np.linalg.norm(line1_start-line2_end)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return np.linalg.norm(line1_end-line2_start)
                return np.linalg.norm(line1_end-line2_end)
                    
                    
            # Segments overlap, return distance between parallel segments
            return np.linalg.norm(((d0*_A)+line1_start)-line2_start)
            
        
        # Lines criss-cross: Calculate the projected closest points
        t = (line2_start - line1_start);
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA/denom;
        t1 = detB/denom;

        pA = line1_start + (_A * t0) # Projected closest point on segment A
        pB = line2_start + (_B * t1) # Projected closest point on segment B


        if t0 < 0:
            pA = line1_start
        elif t0 > magA:
            pA = line1_end
        
        if t1 < 0:
            pB = line2_start
        elif t1 > magB:
            pB = line2_end
            
        # Clamp projection A
        if (t0 < 0) or (t0 > magA):
            dot = np.dot(_B,(pA-line2_start))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = line2_start + (_B * dot)
    
        # Clamp projection B
        if (t1 < 0) or (t1 > magB):
            dot = np.dot(_A,(pB-line1_start))
            if dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = line1_start + (_A * dot)


        return np.linalg.norm(pA-pB)

    def merge_lines(self, lines, angle_tolerance=5.0, distance_tolerance=5.0):
        '''
        Merge two line segments if they are on same line and close enough to each other

        args:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
            angle_tolerance: degree
            distance_tolerance: pixel
        return:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
        '''
        line_count = lines.shape[0]
        old_line_count = 0

        # Loop until line_count not change after merge  
        while old_line_count != line_count:

            X1 = lines[:,0]
            X2 = lines[:,2]
            Y1 = lines[:,1]
            Y2 = lines[:,3]

            deltaX = X2 - X1
            deltaY = Y2 - Y1

            slopes = deltaY / deltaX
            # Change inf slope with a big number
            slopes[slopes == -inf] = -999999
            slopes[slopes == +inf] = +999999

            for i in range(lines.shape[0]):
                for j in range(i, lines.shape[0]):
                    if i == j:
                        # Pass if two lines same
                        continue
                    if self.euclidean_dist(*lines[j]) == 0:
                        # Pass if line deleted
                        continue

                    # Calc distance between lines. If angle is too small we can consider they are in same direction
                    tan_between_lines = (slopes[i] - slopes[j]) / (1 + (slopes[i] * slopes[j]))
                    degree_between_lines = abs(math.degrees(math.atan(tan_between_lines)))

                    if degree_between_lines < angle_tolerance:
                        
                        # Calculate ddistance between two line segments
                        # If two line close enough then merge them

                        line1_p1 = np.array([lines[i][0], lines[i][1], 0])
                        line1_p2 = np.array([lines[i][2], lines[i][3], 0])
                        line2_p1 = np.array([lines[j][0], lines[j][1], 0])
                        line2_p2 = np.array([lines[j][2], lines[j][3], 0])

                        min_distance = self.calc_min_distance(line1_p1, line1_p2, line2_p1, line2_p2)
            
                        if min_distance > distance_tolerance:
                            # If two lines for from each other then do not merge
                            continue

                        line1_norm = self.euclidean_dist(lines[i][0], lines[i][1], lines[i][2], lines[i][3])
                        line2_norm = self.euclidean_dist(lines[j][0], lines[j][1], lines[j][2], lines[j][3])
                            
                        if line1_norm > line2_norm:
                            lines[j] = [0,0,0,0]  # Delete second line
                            slopes[j] = 0  # Delete second line
                        else:
                            lines[i] = lines[j]
                            slopes[i] = slopes[j]
                            lines[j] = [0,0,0,0]  # Delete second line
                            slopes[j] = 0  # Delete second line
            
            # Remove deleted lines
            slopes = slopes[slopes != 0]
            lines = lines[lines[:,3] != 0]

            # Store line counts
            old_line_count = line_count
            line_count = lines.shape[0]
        
        return lines

    def get_line_bounding_box(self, lines, angle_tolerance=5.0, distance_tolerance=5.0):
        '''
        Return line bounding box. Use two lines to create this box if that two lines close enough and has same slope

        Note: This function uses same logic with def merge_lines
                But we try to find 4 point of the bounding box this time

        args:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
            angle_tolerance: degree
            distance_tolerance: pixel
        return:
            boxes: 1D array like [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), ...]
        '''

        boxes = []

        X1 = lines[:,0]
        X2 = lines[:,2]
        Y1 = lines[:,1]
        Y2 = lines[:,3]

        deltaX = X2 - X1
        deltaY = Y2 - Y1

        slopes = deltaY / deltaX
        # Change inf slope with a big number
        slopes[slopes == -inf] = -999999
        slopes[slopes == +inf] = +999999

        for i in range(lines.shape[0]):
            for j in range(i, lines.shape[0]):
                if i == j:
                    # Pass if two lines same
                    continue
                if self.euclidean_dist(*lines[j]) == 0:
                    # Pass if line deleted
                    continue

                # Calc distance between lines. If angle is too small we can consider they are in same direction
                tan_between_lines = (slopes[i] - slopes[j]) / (1 + (slopes[i] * slopes[j]))
                degree_between_lines = abs(math.degrees(math.atan(tan_between_lines)))

                if degree_between_lines < angle_tolerance:
                    
                    # Calculate ddistance between two line segments
                    # If two line close enough then create box

                    line1_p1 = np.array([lines[i][0], lines[i][1], 0])
                    line1_p2 = np.array([lines[i][2], lines[i][3], 0])
                    line2_p1 = np.array([lines[j][0], lines[j][1], 0])
                    line2_p2 = np.array([lines[j][2], lines[j][3], 0])

                    min_distance = self.calc_min_distance(line1_p1, line1_p2, line2_p1, line2_p2)
        
                    if min_distance > distance_tolerance:
                        # If two lines for from each other then do create box
                        continue

                    cnt = np.array([[lines[i][0], lines[i][1]],
                        [lines[j][0], lines[j][1]],
                        [lines[i][2], lines[i][3]],
                        [lines[j][2], lines[j][3]]
                    ])

                    boxes.append(cnt)   
        
        return boxes

    def filter_line(self, image, line_bounding_box, black_thresh):
        '''
        Check the detected line is real or not
        Try to calculate black pixel area ratio inside the bounding box
        Return True if there are really a line inside box
        args:
            image: 2D array standart gray image
            boxes: 1D array like [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), ...]
            black_thresh: int (0-255)
        return:
            result: binary
        '''
        # To search fast we try to find biggest bounding box
        # so we wont check every pixel that inside or not
        # we only check the pixels in the biggest bounding box
        # To find biggest bounding box we have to find max and min x,y for all of the 4 point

        np_arr = np.array(line_bounding_box)
        X = np_arr[:,0]
        Y = np_arr[:,1]

        max_x = max(X)
        min_x = min(X)
        max_y = max(Y)
        min_y = min(Y)

        pixels_inside_contour = []
        # Check the pixels in the image is in inside the box area
        # Slice the image with biggest bounding box
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if cv2.pointPolygonTest(line_bounding_box, (x, y), False) >= 0:
                    pixels_inside_contour.append([x,y])

        # If no pixels found inside contour return false
        if len(pixels_inside_contour) == 0:
            return False

        # Calculate white intensity
        intensity = 0
        for x,y in pixels_inside_contour:
            pixel = image[y][x]
            intensity += pixel

        # mean
        intensity /= len(pixels_inside_contour)

        # Filter intensity
        return (intensity < black_thresh)

    def sort_points(self, points):
        '''
        Sort given points list by calculating their distance to 0,0 point
        0,0 will always work because line points can not be negative in this system

        args:
            points: 2D array like [[pX, pY], [pX, pY], ...]
        return:
            points: 2D array like [[pX, pY], [pX, pY], ...]
        '''
        for i in range(len(points)):
            pX, pY = points[i]
            distance = self.euclidean_dist(0, 0, pX, pY)

            points[i].append(distance)
        
        # Sort points by distance
        points.sort(key = lambda x: x[2])

        points = [[pX, pY] for pX, pY, dist in points]

        return points

    def split_lines_from_intersects(self, lines, intersect_points):
        '''
        Split lines from intersect points
        Resolution tolerance 0.7
        args:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
            intersect_points: 2D array like [[pX, pY], [pX, pY], ...]
        return:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
        '''
        X1 = lines[:,0]
        X2 = lines[:,2]
        Y1 = lines[:,1]
        Y2 = lines[:,3]

        deltaX = X2 - X1
        deltaY = Y2 - Y1

        slopes = deltaY / deltaX

        # We use a 3D list to store satisfying isect points for every line
        # We also store start and end points for that line with isects
        lines_isect_points = []

        for i in range(lines.shape[0]):
            isect_points_for_current_line = []
            for j in range(len(intersect_points)):
                # Line Eq = y - y1 = m(x - x1)
                # If a intersect point satisfying the equation then we can think that
                # this Intersect point is on the line and we use it to split the line

                isectX = intersect_points[j][0]
                isectY = intersect_points[j][1]

                # We use start, end point and slope to create line eq
                line_startX = lines[i][0]
                line_startY = lines[i][1]
                line_endX = lines[i][2]
                line_endY = lines[i][3]

                slope = slopes[i]

                # If slope is going to inf then Eq will be x = c 
                if abs(slope) == inf:
                    # Check x values are same and y between line segment
                    if isectX == line_startX and ((line_startY < isectY < line_endY) or (line_startY > isectY > line_endY)):
                        isect_points_for_current_line.append([isectX, isectY])

                # If slope is going to 0 then Eq will be y = c 
                elif abs(slope) <= 0.001:
                    # Check y values are same and x between line segment
                    if isectY == line_startY and ((line_startX < isectX < line_endX) or (line_startX > isectX > line_endX)):
                        isect_points_for_current_line.append([isectX, isectY])

                else:
                    # y = m(x - x1) + y1
                    possibleY = (slope * (isectX - line_startX)) + line_startY

                    # x = (y - y1)/m + x1
                    possibleX = ((isectY - line_startY)/slope) + line_startX

                    if (isectY < possibleY + 0.7 and isectY > possibleY - 0.7) or (isectX < possibleX + 0.7 and isectX > possibleX - 0.7):
                        isect_points_for_current_line.append([isectX, isectY])


            line_start = [lines[i][0], lines[i][1]]
            line_end = [lines[i][2], lines[i][3]]

            lines_isect_points.append([line_start] + isect_points_for_current_line + [line_end])

        # Split lines
        new_lines = []

        for i in range(lines.shape[0]):
            # Sort points by their distance to 0,0 point
            lines_isect_points[i] = self.sort_points(lines_isect_points[i])
            # Get lines between sorted isec points
            for j in range(len(lines_isect_points[i]) - 1):
                startX = lines_isect_points[i][j][0]
                startY = lines_isect_points[i][j][1]
                endX = lines_isect_points[i][j+1][0]
                endY = lines_isect_points[i][j+1][1]

                new_line = [startX, startY, endX, endY]

                new_lines.append(new_line)
        
        return np.array(new_lines)

    def lines_to_dict(self, lines):
        '''
        Transform line list to dictionary with adding orientations
        args:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
        return:
            lines: dictionary like {'line1': [spX, spY, epX, epY], 'line2': [spX, spY, epX, epY], ...}
        '''
        X1 = lines[:,0]
        X2 = lines[:,2]
        Y1 = lines[:,1]
        Y2 = lines[:,3]

        deltaX = X2 - X1
        deltaY = Y2 - Y1

        slopes = deltaY / deltaX

        orinetations = []
        for slope in slopes:
            if slope > 28 or slope < -28:  # More then 88 degree
                orinetations.append('Vrt')
            elif -0.3 < slope < 0.3 :  # Means between then -2 and 2 degree
                orinetations.append('Hrz')
            else:
                orinetations.append('Tlt')

        line_dict = {}
        for i in range(len(lines)):
            line_dict['line'+str(i)] = [float(line) for line in lines[i]] + [orinetations[i]]

        return line_dict


    def intersec_to_dict(self, splitted_lines_dict, intersections):
        '''
        Transform intersection list to dictionary with lines
        Resolution tolerance 0.7
        args:
            splitted_lines_dict: dictionary like {'line1': [spX, spY, epX, epY], 'line2': [spX, spY, epX, epY], ...}
            intersect_points: 2D array like [[pX, pY], [pX, pY], ...]
        return:
            intersect_dict: dictionary like {'isect1': [[x, y], [line12, line35]], ...}
        '''
        intersect_dict = {}
        count = 0
        for i in range(len(intersections)):
            lines_touched_isect = []
            pX, pY = intersections[i]
            for key, value in splitted_lines_dict.items():
                spX, spY, epX, epY, orient = value
                if (spX - 0.7 < pX < spX + 0.7 and spY - 0.7 < pY < spY+ 0.7) or (epX - 0.7 < pX < epX + 0.7 and epY - 0.7 < pY < epY+ 0.7):
                    lines_touched_isect.append(key)

            intersect_dict['isect' + str(i)] = [[pX, pY], lines_touched_isect]

        return intersect_dict


    def adder_function(self, intersect_dict, new_intersects):
        '''
        Used to append new intersects to old ones
        args:
            intersect_dict: dictionary like {'isect1': [[x, y], [line12, line35]], ...}
            new_intersects: dictionary like {'isect1': [[x, y], [line12, line35]], ...}
        return:
            intersect_dict: dictionary like {'isect1': [[x, y], [line12, line35]], ...}
        '''
        key_list = list(intersect_dict.keys())
        key_numbers = [int(key[5:]) for key in key_list]
        key_numbers.sort()  # Sort key numbers

        last_key_num = key_numbers[-1]
        new_key_num = last_key_num + 1

        for key, value in new_intersects.items():
            intersect_dict['isect' + str(new_key_num)] = value
            new_key_num += 1

        return intersect_dict


    def extend_lines(self, lines, max_extend):
        '''
        Extend lines little
        
        Max Extend is maximum extension length. To protect ratio other axis extension length calculated from this number 
        args:
            max_extend: pixel
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
        return:
            lines: 2D array like [[spX, spY, epX, epY], [spX, spY, epX, epY], ...]
        '''
        for i in range(lines.shape[0]):
            line_startX = lines[i][0]
            line_startY = lines[i][1]
            line_endX = lines[i][2]
            line_endY = lines[i][3]

            deltaX = line_endX - line_startX
            deltaY = line_endY - line_startY

            slope = deltaY / deltaX

            if slope > 28:  # More then 88 degree
                extendX = 0
                extendY = max_extend

                if line_endY > line_startY:
                    lines[i][1] -= extendY
                    lines[i][3] += extendY
                else:
                    lines[i][1] += extendY
                    lines[i][3] -= extendY

            elif slope > 0.3:
                deltaX = abs(deltaX)
                deltaY = abs(deltaY)
                if deltaX > deltaY:
                    extendX = max_extend  # extend 2px x axis
                    extendY = deltaY * extendX / deltaX  # extend y axis in same rate
                else:
                    extendY = max_extend  # extend 2px y axis
                    extendX = deltaX * extendY / deltaY  # extend x axis in same rate 

                if line_endX > line_startX:
                    lines[i][0] -= extendX
                    lines[i][2] += extendX
                else:
                    lines[i][0] += extendX
                    lines[i][2] == extendX

                if line_endY > line_startY:
                    lines[i][1] -= extendY
                    lines[i][3] += extendY
                else:
                    lines[i][1] += extendY
                    lines[i][3] -= extendY

            elif -0.3 < slope < 0.3 :  # Means between then -2 and 2 degree
                extendX = max_extend
                entendY = 0

                if line_endX > line_startX:
                    lines[i][0] -= extendX
                    lines[i][2] += extendX
                else:
                    lines[i][0] += extendX
                    lines[i][2] -= extendX
                
            elif slope < 0.3:
                deltaX = abs(deltaX)
                deltaY = abs(deltaY)
                if deltaX > deltaY:
                    extendX = max_extend  # extend 2px x axis
                    extendY = deltaY * extendX / deltaX  # extend y axis in same rate
                else:
                    extendY = max_extend  # extend 2px y axis
                    extendX = deltaX * extendY / deltaY  # extend x axis in same rate 

                if line_endX > line_startX:
                    lines[i][0] -= extendX
                    lines[i][2] += extendX
                else:
                    lines[i][0] += extendX
                    lines[i][2] -= extendX

                if line_endY > line_startY:
                    lines[i][1] -= extendY
                    lines[i][3] += extendY
                else:
                    lines[i][1] += extendY
                    lines[i][3] -= extendY

            elif slope < -28:  # Less then 88 degree
                extendX = 0
                entendY = max_extend

                if line_endY > line_startY:
                    lines[i][1] -= extendY
                    lines[i][3] += extendY
                else:
                    lines[i][1] += extendY
                    lines[i][3] -= extendY

        return lines
