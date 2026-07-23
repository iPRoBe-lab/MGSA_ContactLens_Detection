# code adapted from: https://github.com/andreibercu/iris-recognition/tree/master
import numpy as np
import cv2
import math
import random

def get_iris_boundaries(img, show=False):
    # Finding iris inner boundary
    pupil_circle = find_pupil(img)

    if not pupil_circle:
        print('ERROR: Pupil circle not found!')
        return None, None

    # Finding iris outer boundary
    radius_range = int(math.ceil(pupil_circle[2]*1.5))
    multiplier = 0.25
    center_range = int(math.ceil(pupil_circle[2]*multiplier)) 
    ext_iris_circle = find_ext_iris(
                        img, pupil_circle, center_range, radius_range)

    while(not ext_iris_circle and multiplier <= 0.7):
        multiplier += 0.05
        print('Searching exterior iris circle with multiplier ' + \
              str(multiplier))
        center_range = int(math.ceil(pupil_circle[2]*multiplier))
        ext_iris_circle = find_ext_iris(img, pupil_circle,
                                        center_range, radius_range)
    if not ext_iris_circle:
        print('ERROR: Exterior iris circle not found!')
        return None, None
    
    if show:
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        draw_circles(cimg, pupil_circle, ext_iris_circle,
                     center_range, radius_range)
        cv2.imshow('iris boundaries', cimg)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pupil_circle, ext_iris_circle

def find_pupil(img):
    def get_edges(image):
        edges = cv2.Canny(image,20,100)
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        ksize = 2 * random.randrange(5,11) + 1
        edges = cv2.GaussianBlur(edges,(ksize,ksize),0)
        return edges

    param1 = 200 # 200
    param2 = 120 # 150
    pupil_circles = []
    while(param2 > 35 and len(pupil_circles) < 10):
        for mdn,thrs in [(m,t)
                         for m in [3,5,7]
                         for t in [20,25,30,35,40,45,50,55,60]]:
            # Median Blur
            median = cv2.medianBlur(img, 2*mdn+1)

            # Threshold
            ret, thres = cv2.threshold(
                            median, thrs, 255,
                            cv2.THRESH_BINARY_INV)

            # Fill Contours
            # contours, hierarchy = \
            #         cv2.findContours(thres.copy(),
            #                          cv2.RETR_EXTERNAL,
            #                          cv2.CHAIN_APPROX_NONE)
            # draw_con = cv2.drawContours(thres, contours, -1, (255), -1)

            # Canny Edges
            edges = get_edges(thres)

            # HoughCircles
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1,
                                       np.array([]), param1, param2)                
            if circles is not None and len(circles) > 0:
                # convert the (x, y) coordinates and radius of the circles 
                # to integers
                circles = np.round(circles[0, :]).astype("int")
                for c in circles:
                    pupil_circles.append(c)

        param2 = param2 -10

    # cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    return get_mean_circle(pupil_circles)

def get_mean_circle(circles, draw=None):
    if not circles:
        return
    mean_0 = int(np.mean([c[0] for c in circles]))
    mean_1 = int(np.mean([c[1] for c in circles]))
    mean_2 = int(np.mean([c[2] for c in circles]))

    if draw is not None:
        draw = draw.copy()
        # draw the outer circle
        cv2.circle(draw,(mean_0,mean_1),mean_2,(0,255,0),1)
        # draw the center of the circle
        cv2.circle(draw,(mean_0,mean_1),2,(0,255,0),2)
        cv2.imshow('mean circle', draw)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows() 
    
    return mean_0, mean_1, mean_2
    
def find_ext_iris(img, pupil_circle, center_range, radius_range):
    def get_edges(image, thrs2):
        thrs1 = 0 # 0
        edges = cv2.Canny(image, thrs1, thrs2, apertureSize=5)
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        ksize = 2 * random.randrange(5,11) + 1
        edges = cv2.GaussianBlur(edges,(ksize,ksize),0)
        return edges

    def get_circles(hough_param, median_params, edge_params):
        crt_circles = []
        for mdn,thrs2 in [(m,t)
                          for m in median_params
                          for t in edge_params]:
            # Median Blur
            median = cv2.medianBlur(img, 2*mdn+1)

            # Canny Edges
            edges = get_edges(median, thrs2)

            # HoughCircles
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1,
                                       np.array([]), 200, hough_param)                
            if circles is not None and len(circles) > 0:
                # convert the (x, y) coordinates and radius of the 
                # circles to integers
                circles = np.round(circles[0, :]).astype("int")
                for (c_col, c_row, r) in circles:
                    if point_in_circle(
                            int(pupil_circle[0]), int(pupil_circle[1]),
                            center_range, c_col, c_row) and \
                       r > radius_range:
                        crt_circles.append((c_col, c_row, r))
        return crt_circles

    param2 = 120 # 150
    total_circles = []
    while(param2 > 40 and len(total_circles) < 10):
        crt_circles = get_circles(
                        param2, [8,10,12,14,16,18,20], [430,480,530])
        if crt_circles:
            total_circles += crt_circles
        param2 = param2 -10

    if not total_circles:
        print("Running plan B on finding ext iris circle")
        param2 = 120
        while(param2 > 40 and len(total_circles) < 10):
            crt_circles = get_circles(
                            param2, [3,5,7,21,23,25], [430,480,530])
            if crt_circles:
                total_circles += crt_circles
            param2 = param2 -10

    if not total_circles:
        return

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    filtered = filtered_circles(total_circles)

    return get_mean_circle(filtered)

def point_in_circle(c_col, c_row, c_radius, p_col, p_row):
    return distance(c_col, c_row, p_col, p_row) <= c_radius
    
def filtered_circles(circles, draw=None):
    # what if there are only 2 circles - which is alpha?
    def get_alpha_radius(circles0):
        alpha_circle = None
        dist_min = None
        circles1 = circles0[:]
        circles2 = circles0[:]
        for crt_c in circles1:
            dist = 0
            for c in circles2:
                dist += math.fabs(float(crt_c[2]) - float(c[2]))
            if not dist_min or dist < dist_min:
                dist_min = dist
                alpha_circle = crt_c
        return alpha_circle[2]

    if not circles:
        print('Error: empty circles list in filtered_circles() !')
        return []
    c_0_mean, c_0_dev = standard_dev([int(i[0]) for i in circles])
    c_1_mean, c_1_dev = standard_dev([int(i[1]) for i in circles])
    filtered = []
    filtered_pos = []
    not_filtered = []
    ratio = 1.5 
    for c in circles[:]:
        if c[0] < c_0_mean - ratio*c_0_dev or \
           c[0] > c_0_mean + ratio*c_0_dev or \
           c[1] < c_1_mean - ratio*c_1_dev or \
           c[1] > c_1_mean + ratio*c_1_dev:
            not_filtered.append(c)
        else:
            filtered_pos.append(c)
    if len([float(c[2]) for c in filtered_pos]) < 3:
        filtered = filtered_pos
    else:
        alpha_radius = get_alpha_radius(filtered_pos)
        mean_radius, dev_radius = standard_dev(
                                    [float(c[2]) for c in filtered_pos])
        max_radius = alpha_radius + dev_radius
        min_radius = alpha_radius - dev_radius
        for c in filtered_pos:
            if c[2] < min_radius or \
               c[2] > max_radius:
                not_filtered.append(c)
            else:
                filtered.append(c)

    if draw is not None:
        draw = draw.copy()
        for circle in not_filtered:
            # draw the outer circle
            cv2.circle(draw,(circle[0],circle[1]),circle[2],(255,0,0),1)
            # draw the center of the circle
            cv2.circle(draw,(circle[0],circle[1]),2,(255,0,0),2)
        for circle in filtered:
            # draw the outer circle
            cv2.circle(draw,(circle[0],circle[1]),circle[2],(0,255,0),1)
            # draw the center of the circle
            cv2.circle(draw,(circle[0],circle[1]),2,(0,255,0),2)
        cv2.imshow('filtered_circles() total={0} filtered_pos={1} filtered={2}'.\
                   format(len(circles), len(filtered_pos), len(filtered)),
                   draw)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()
    return filtered

def draw_circles(cimg, pupil_circle, ext_iris_circle,
                 center_range=None, radius_range=None):
    # draw the outer pupil circle
    cv2.circle(cimg,(pupil_circle[0], pupil_circle[1]), pupil_circle[2],
                     (0,0,255),1)
    # draw the center of the pupil circle
    cv2.circle(cimg,(pupil_circle[0],pupil_circle[1]),1,(0,0,255),1)
    if center_range:
        # draw ext iris center range limit
        cv2.circle(cimg,(pupil_circle[0], pupil_circle[1]), center_range,
                         (0,255,255),1)
    if radius_range:
        # draw ext iris radius range limit
        cv2.circle(cimg,(pupil_circle[0], pupil_circle[1]), radius_range,
                         (0,255,255),1)
    # draw the outer ext iris circle
    cv2.circle(cimg, (ext_iris_circle[0], ext_iris_circle[1]), 
               ext_iris_circle[2],(0,255,0),1)
    # draw the center of the ext iris circle
    cv2.circle(cimg, (ext_iris_circle[0], ext_iris_circle[1]), 
               1,(0,255,0),1)
    
def get_equalized_iris(img, ext_iris_circle, pupil_circle, show=False):
    def find_roi():
        mask = img.copy()
        mask[:] = (0)

        cv2.circle(mask, 
                   (ext_iris_circle[0], ext_iris_circle[1]), 
                   ext_iris_circle[2], (255), -1)
        cv2.circle(mask,
                   (pupil_circle[0],pupil_circle[1]),
                   pupil_circle[2],(0), -1)

        roi = cv2.bitwise_and(img, mask)

        return roi

    roi = find_roi()

    # Mask the top side of the iris
    for p_col in range(roi.shape[1]):
        for p_row in range(roi.shape[0]):
            theta = angle_v(ext_iris_circle[0], ext_iris_circle[1], 
                            p_col, p_row)
            if theta > 50 and theta < 130:
                roi[p_row,p_col] = 0

    ret, roi = cv2.threshold(roi,50,255,cv2.THRESH_TOZERO)

    equ_roi = roi.copy()
    cv2.equalizeHist(roi, equ_roi)
    roi = cv2.addWeighted(roi, 0.0, equ_roi, 1.0, 0)

    if show:
        cv2.imshow('equalized histogram iris region', roi)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return roi
    

def angle_v(x1,y1,x2,y2):
    return math.degrees(math.atan2(-(y2-y1),(x2-x1)))

def distance(x1,y1,x2,y2):
    dst = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dst

def mean(x):
    # sum = 0.0
    # for i in range(len(x)):
    #     sum += x[i]
    # return sum/len(x)
    return np.mean(x)

def median(x):
    return np.median(x)

# def standard_dev(x):
#     if not x:
#         print('Error: empty list parameter in standard_dev() !')
#         return None, None
#     m = mean(x)
#     sumsq = 0.0
#     for i in range(len(x)):
#         sumsq += (x[i] - m) ** 2
#     return m, math.sqrt(sumsq/len(x))

def standard_dev(x):
    if not x:
        print('Error: empty list parameter in standard_dev() !')
        return None, None
    x = np.array(x)
    return np.mean(x), np.std(x)
    


