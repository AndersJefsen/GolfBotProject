from windowcapture import WindowCapture
from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
from ultralytics import YOLO
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision 


import torch

def detect_objects(imageToDetectOn, imageToDrawOn, hsv_filter, maxThreshold,minThreshold,minArea,maxArea,name,rgb_Color):
    #this hsv filter is used to find the edges of the obstacles
    
    img = vision_image.apply_hsv_filter(imageToDetectOn, hsv_filter)
    h, s, v = cv.split(img)
  
    imgray = v
    #the strength of the edge detection where edxes between are considered weak edges and
    # are only shown if they are connected to strong edges which is above maxVal
    maxVal = maxThreshold
    minVal = minThreshold
    minArea = minArea
    maxArea = maxArea
    # Find Canny edges 
    edged = cv.Canny(imgray,minVal, maxVal) 
    contours, hierarchy = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea > area > minArea:
            cv.drawContours(imageToDrawOn, cnt, -1, rgb_Color, 3) 
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
          
            x, y, w, h = cv.boundingRect(approx)
            #cv.rectangle(imageToDrawOn, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv.putText(imageToDrawOn, name, (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, .7, rgb_Color, 2)
            cv.putText(imageToDrawOn, "Points: " + str(len(approx)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, .7, rgb_Color, 2)
            cv.putText(imageToDrawOn, "Area: " + str(int(area)), (x + w + 20, y + 70), cv.FONT_HERSHEY_COMPLEX, .7, rgb_Color, 2)
            for point in approx:
                cv.circle(imageToDrawOn, tuple(point[0]), 5, (255, 0, 0), 3)  # Adjust circle radius

    
    
    return edged, imageToDrawOn

def findRoundObjects(imageToDetectOn, imageToDrawOn):
       #this hsv filter is used to find the edges of the obstacles
    #hsv_filter = HsvFilter(0, 155, 0, 179, 255, 255, 0, 0, 0, 0)
    #img = vision_image.apply_hsv_filter(screenshot, hsv_filter)
    

    h, s, v = cv.split(imageToDetectOn)
    #imgBlur = cv.GaussianBlur(imageToFindObstaclesIn, (7, 7), 1)
    #cv.imshow('blur', imgBlur)
    #so i proberly allready have the gray scale image

    
    imgray = v


    blur = cv.GaussianBlur(imgray, (7, 7), cv.BORDER_DEFAULT)
    
    thresholdmin = cv.getTrackbarPos('threshold min', 'Trackbars')
    thresholdmax = cv.getTrackbarPos('threshold max', 'Trackbars')
    _, threshold = cv.threshold(blur, thresholdmin,thresholdmax, cv.THRESH_BINARY)

    #imgray = cv.cvtColor(screenshot, cv.COLOR_HSV2GRAY)
    rzb = cv.resize(threshold, (960, 540))
    cv.imshow('thresh', rzb)
    
    #the strength of the edge detection where edxes between are considered weak edges and
    # are only shown if they are connected to strong edges which is above maxVal
    maxVal = 192
    minVal = 150
    

    threshold1 = cv.getTrackbarPos('threshold1', 'Trackbars')
    threshold2 = cv.getTrackbarPos('threshold2', 'Trackbars')
    minArea = cv.getTrackbarPos('minArea', 'Trackbars')
    maxArea = cv.getTrackbarPos('maxArea', 'Trackbars')

    #minArea = 15000
    # Find Canny edges 
    edged = cv.Canny(threshold,threshold1, threshold2) 


   # kernel = np.ones((5,5))
   # dilation = cv.dilate(edged,kernel,iterations = 1)
   # closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
   #imgDil = cv.dilate(edged, kernel, iterations=1)


    contours, hierarchy = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea > area > minArea:
            cv.drawContours(imageToDrawOn, cnt, -1, (0, 255, 0), 3) 
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
          
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(imageToDrawOn, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv.putText(imageToDrawOn, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv.putText(imageToDrawOn, "Area: " + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            for point in approx:
                cv.circle(imageToDrawOn, tuple(point[0]), 5, (255, 0, 0), 3)  # Adjust circle radius

    #https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    #removes noise from image 
    #closing = cv.morphologyEx(imgray, cv.MORPH_CLOSE, kernel)
    #opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    # Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
    
    
    return edged, output_image

def findArena(imageToDetectOn, imageToDrawOn): 
    hsv_filter = HsvFilter(0, 104, 0, 179, 255, 255, 0, 0, 0, 0)
    img = vision_image.apply_hsv_filter(imageToDetectOn, hsv_filter)
    h, s, v = cv.split(img)
    imgray = v
    blur = cv.GaussianBlur(imgray, (7, 7), cv.BORDER_DEFAULT)
    thresholdmin = 44
    thresholdmax = 255
    _, threshold = cv.threshold(blur, thresholdmin,thresholdmax, cv.THRESH_BINARY)
    threshold1 = 100
    threshold2 = 200
    edged = cv.Canny(threshold,threshold1, threshold2) 
    contours, hierarchy = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    minArea = 2435
    maxArea = 357513

    points = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea > area > minArea:
            cv.drawContours(imageToDrawOn, cnt, -1, (0, 0, 0), 3) 
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
          
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(imageToDrawOn, (x, y), (x + w, y + h), (0, 0, 0), 5)
            cv.putText(imageToDrawOn, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, .7, (0, 0, 0), 2)
            cv.putText(imageToDrawOn, "Area: " + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, .7, (0, 0, 0), 2)
            for point in approx:
                cv.circle(imageToDrawOn, tuple(point[0]), 5, (0, 0, 0), 3)  # Adjust circle radius
                points.append(point[0])
    
    if points:
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(imageToDetectOn.shape[:2], dtype=np.uint8)
        cv.fillPoly(mask, [points], 255)

        # Invert the mask
        inverted_mask = cv.bitwise_not(mask)

        # Create black background for outside region
        black_background = np.zeros_like(imageToDetectOn)

        # Isolate the inside region
        inside_region = cv.bitwise_and(imageToDetectOn, imageToDetectOn, mask=mask)

        # Isolate the outside region
        outside_region = cv.bitwise_and(black_background, black_background, mask=inverted_mask)

        # Combine inside and outside regions
        final_image = cv.add(inside_region, outside_region)

        rzb = cv.resize(final_image, (960, 540))
        cv.imshow('masked', rzb)
        return edged, final_image
    else:
        print("No points found.")
        return edged, imageToDrawOn

def findRobot(imageToDetectOn, imageToDrawOn):
    #this hsv filter is used to find the edges of the obstacles
    hsv_filter = HsvFilter(98, 74, 0, 145, 255, 255, 0, 0, 0, 0)
    maxVal = 200
    minVal = 100
    minArea = 259
    maxArea = 10000
    color = (0, 42, 255)
    name = "robot"
    
    return detect_objects(imageToDetectOn, imageToDrawOn, hsv_filter, maxVal, minVal, minArea, maxArea, name, color)

def findCross(imageToDetectOn, imageToDrawOn):
    hsv_filter = HsvFilter(0, 96, 137, 179, 255, 227, 0, 0, 0, 0)
    maxVal = 0
    minVal = 0
    minArea = 3539
    maxArea = 5181
    color = (157, 50, 168)
    name = "cross"

    return detect_objects(imageToDetectOn, imageToDrawOn, hsv_filter, maxVal, minVal, minArea, maxArea, name, color)
#still missing folowing functions
def findOrangeBall(imageToDetectOn, imageToDrawOn):
    hsv_filter = HsvFilter(0, 107, 209, 179, 255, 255, 0, 0, 0, 0)
    maxVal = 100
    minVal = 200
    minArea = 311
    maxArea = 1000
    color = (255, 0, 200)
    name = "orange ball"

    return detect_objects(imageToDetectOn, imageToDrawOn, hsv_filter, maxVal, minVal, minArea, maxArea, name, color)

#def findWhiteBall(imageToDetectOn, imageToDrawOn):
#def findEgg(imageToDetectOn, imageToDrawOn):
#def findArena(imageToDetectOn, imageToDrawOn):

def findObstacles(screenshot, output_image):
    #this hsv filter is used to find the edges of the obstacles
    #hsv_filter = HsvFilter(0, 155, 0, 179, 255, 255, 0, 0, 0, 0)
    #img = vision_image.apply_hsv_filter(screenshot, hsv_filter)
    

    h, s, v = cv.split(screenshot)
    #imgBlur = cv.GaussianBlur(imageToFindObstaclesIn, (7, 7), 1)
    #cv.imshow('blur', imgBlur)
    #so i proberly allready have the gray scale image

    
    imgray = v
    #imgray = cv.cvtColor(screenshot, cv.COLOR_HSV2GRAY)
   
    
    #the strength of the edge detection where edxes between are considered weak edges and
    # are only shown if they are connected to strong edges which is above maxVal
    maxVal = 192
    minVal = 150

    threshold1 = cv.getTrackbarPos('threshold1', 'Trackbars')
    threshold2 = cv.getTrackbarPos('threshold2', 'Trackbars')
    minArea = cv.getTrackbarPos('minArea', 'Trackbars')
    maxArea = cv.getTrackbarPos('maxArea', 'Trackbars')
    #minArea = 15000
    # Find Canny edges 
    edged = cv.Canny(imgray,threshold1, threshold2) 


   # kernel = np.ones((5,5))
   # dilation = cv.dilate(edged,kernel,iterations = 1)
   # closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
   #imgDil = cv.dilate(edged, kernel, iterations=1)


    contours, hierarchy = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea > area > minArea:
            cv.drawContours(output_image, cnt, -1, (0, 255, 0), 3) 
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
          
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv.putText(output_image, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv.putText(output_image, "Area: " + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            for point in approx:
                cv.circle(output_image, tuple(point[0]), 5, (255, 0, 0), 3)  # Adjust circle radius

    #https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    #removes noise from image 
    #closing = cv.morphologyEx(imgray, cv.MORPH_CLOSE, kernel)
    #opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    # Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
    
    
    return edged, output_image

def createMask(imageToDetectOn,points):
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros(imageToDetectOn.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [points], 255)

    return mask

def useMask(imageToMask,mask):
    return cv.bitwise_and(imageToMask, imageToMask, mask=mask)
#model = YOLO('best.pt')
# initialize the WindowCapture class


#wincap = WindowCapture(None)    
arenaCorners = []
mask = None
wincap = cv.VideoCapture(0,cv.CAP_DSHOW)

vision_image = Vision('ball.png')

vision_image.init_control_gui()

findArena = False
while not findArena:
    try:
        ret, screenshot = wincap.read()
        output_image = screenshot.copy()
        if screenshot is None:
            print("Failed to capture screenshot.")
            continue
        findArena, output_image,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ComputerVision.ImageProcessor.find_Arena(screenshot, output_image)
        if findArena:
            arenaCorners.append(bottom_left_corner)
            arenaCorners.append(bottom_right_corner)
            
            arenaCorners.append(top_right_corner)
            arenaCorners.append(top_left_corner)
            mask = createMask(screenshot,arenaCorners)
            print("mask Created")
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

        

loop_time = time()
while(True):
    try:
        #screenshot = wincap.get_screenshot()
        ret, screenshot = wincap.read()
        if screenshot is None:
            print("Failed to capture screenshot.")
            continue

        inputimg = useMask(screenshot,mask)
        

        #cross = ComputerVision.find_cross_contours(screenshot)    
        #outputhsv_image = vision_image.apply_hsv_filter(screenshot)

        #outputedge_image = vision_image.apply_edge_filter(outputhsv_image)
        #rectangles = vision_image.find(outputhsv_image,0.5,10)
        
        #ret, thresh = cv.threshold(outputedge_image, 127, 255, 0)
        #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #output_image = cv.drawContours(screenshot, contours, -1, (0,255,0), 3)
         #points are only used for crosshair
         #points = vision_image.get_points(rectangles)
        #output_image = vision_image.draw_rectangles(screenshot, rectangles)
        #output_image = vision_image.draw_crosshairs(output_image, points)
        #inputimg = screenshot
        #edged, output_image = findArena(inputimg,screenshot)
        #outputhsv_image = vision_image.apply_hsv_filter(output_image)
        #edged, output_image = findRobot(inputimg,screenshot)
        #edged, output_image = findOrangeBall(inputimg,screenshot)
        #edged, output_image = findCross(inputimg,screenshot)
        #edged, output_image = findObstacles(outputhsv_image,screenshot)
        #edged, output_image = findRoundObjects(outputhsv_image,screenshot)
        '''
        rzhsv = cv.resize(outputhsv_image, (960, 540))
        cv.imshow('hsv', rzhsv)

        rze = cv.resize(edged, (960, 540))
        cv.imshow('edges', rze)
        '''
       
        screenoutput = cv.resize(inputimg, (960, 540))
        cv.imshow('Computer Vision', screenoutput)

        #screengray = cv.resize(imgray, (960, 540))
        
        #screenhsvimg = cv.resize(outputhsv_image, (960, 540))
        #screenedgeimg = cv.resize(outputedge_image, (960, 540))
        #cv.imshow('gray', screengray)
        #cv.imshow('hsv', screenhsvimg)
        #cv.imshow('edge', screenedgeimg)

      

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

cv.destroyAllWindows()
wincap.release()
print('Done.')



