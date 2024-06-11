
import cv2 as cv
import numpy as np 
import os
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter


class Vision:

    TRACKBAR_WINDOW = "Trackbars"

    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    #constructor
    def __init__(self, needle_img_path, method = cv.TM_CCOEFF_NORMED):
        self.needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        self.method = method


    def find(self, haystack_img,threshold=0.5,max_results=10):
       
        result = cv.matchTemplate(haystack_img, self.needle_img, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        if not locations:
            return np.array([], dtype=np.int32).reshape(0, 4)


        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            #we add the rectangle to the list twice beacuse groupRectangles only looks for groups of 2 and removes observations that are not in a group of 2
            rectangles.append(rect)
            rectangles.append(rect)

        #print(rectangles)
        # makes only one rectangle pr observation
        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
        

        if len(rectangles) > max_results:
            print('Warning: too many results, raise the threshold.')
            rectangles = rectangles[:max_results]

        return rectangles
    


    def get_points(self, rectangles):
        points = []

        for (x,y,w,h) in rectangles:

            center_x = x + int(w/2)
            center_y = y + int(h/2)
            #saving the points
            points.append((center_x,center_y))

        return points
    


   
   
    def draw_rectangles(self, haystack_img, rectangles):
        line_color = (0,255,0)
        line_type = cv.LINE_4
        for (x,y,w,h) in rectangles:
            top_left = (x,y)
            bottom_right = (x+w, y+h)
            cv.rectangle(haystack_img,top_left,bottom_right,line_color, thickness=2, lineType=line_type)
        return haystack_img
   

    def draw_crosshairs(self, haystack_img, points):
        marker_color = (255,0,255)
        marker_type = cv.MARKER_CROSS
        for (center_x, center_y) in points:
            cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)
        return haystack_img

    def init_control_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW,cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW,350,700)


        def nothing(position):
            pass

        # Create the trackbars
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)    
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        # trackbars for increasing/decreasing the value of hue, saturation, and value
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # trackbar for edge creation
        cv.createTrackbar('kernelSize', self.TRACKBAR_WINDOW, 1, 30, nothing)
        cv.createTrackbar('ErodeIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('DilateIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('Canny1', self.TRACKBAR_WINDOW, 0, 200, nothing)
        cv.createTrackbar('Canny2', self.TRACKBAR_WINDOW, 0, 500, nothing)
        #set default value for canny trackbars
        cv.setTrackbarPos('kernelSize', self.TRACKBAR_WINDOW, 5)
        cv.setTrackbarPos('Canny1', self.TRACKBAR_WINDOW, 100)
        cv.setTrackbarPos('Canny2', self.TRACKBAR_WINDOW, 200)

        cv.createTrackbar('threshold1', self.TRACKBAR_WINDOW, 0, 1000, nothing)
        cv.createTrackbar('threshold2', self.TRACKBAR_WINDOW, 0, 1000, nothing)
        cv.setTrackbarPos('threshold1', self.TRACKBAR_WINDOW, 100)
        cv.setTrackbarPos('threshold2', self.TRACKBAR_WINDOW, 200)

        cv.createTrackbar('minArea', self.TRACKBAR_WINDOW, 0, 10000, nothing)
        cv.setTrackbarPos('minArea', self.TRACKBAR_WINDOW, 1000)
        cv.createTrackbar('maxArea', self.TRACKBAR_WINDOW, 0, 1000000, nothing)
        cv.setTrackbarPos('maxArea', self.TRACKBAR_WINDOW, 100000)
        

        cv.createTrackbar('threshold min', self.TRACKBAR_WINDOW, 0, 255, nothing)  
        cv.createTrackbar('threshold max', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.setTrackbarPos('threshold min', self.TRACKBAR_WINDOW, 0) 
        cv.setTrackbarPos('threshold max', self.TRACKBAR_WINDOW, 255)

        cv.createTrackbar('minPoints', self.TRACKBAR_WINDOW, 1, 100, nothing)
        cv.setTrackbarPos('minPoints', self.TRACKBAR_WINDOW, 1)
        cv.createTrackbar('maxPoints', self.TRACKBAR_WINDOW, 1, 100, nothing)
        cv.setTrackbarPos('maxPoints', self.TRACKBAR_WINDOW, 100)
    def get_hsv_filter_from_controls(self):
        # Get current positions of all trackbars
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter
    
    def get_edge_filter_from_controls(self):
        # Get current positions of all trackbars
        edge_filter = EdgeFilter()
        edge_filter.kernelSize = cv.getTrackbarPos('kernelSize', self.TRACKBAR_WINDOW)
        edge_filter.erodeIter = cv.getTrackbarPos('ErodeIter', self.TRACKBAR_WINDOW)
        edge_filter.dilateIter = cv.getTrackbarPos('DilateIter', self.TRACKBAR_WINDOW)
        edge_filter.canny1 = cv.getTrackbarPos('Canny1', self.TRACKBAR_WINDOW)
        edge_filter.canny2 = cv.getTrackbarPos('Canny2', self.TRACKBAR_WINDOW)
        return edge_filter


    def apply_hsv_filter(self, original_image, hsv_filter=None):
        #convert the image to hsv
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # Add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])

        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        #convert back to BGR for imshow
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img
    
    def apply_edge_filter(self, original_image, edge_filter=None):
        if not edge_filter:
            edge_filter = self.get_edge_filter_from_controls()
        
        kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
        eroded_image = cv.erode(original_image, kernel, iterations=edge_filter.erodeIter)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

        result = cv.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

        img = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

        return img

    def shift_channel(self,c,amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c
    