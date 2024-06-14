from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
from path import find_close_ball
import asyncio
import com
import threading
import numpy as np
import sys
import logging
import os
from queue import Queue
from time import time, strftime, gmtime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision 

def detect_objects(imageToDetectOn, imageToDrawOn,vision_image, hsv_filter, maxThreshold,minThreshold,minArea,maxArea,name,rgb_Color,threshold,minPoints,maxPoints,arenaCorners):
       #this hsv filter is used to find the edges of the obstacle
    img = vision_image.apply_hsv_filter(imageToDetectOn, hsv_filter)
    h, s, v = cv.split(img)
    imgray = v
    blur = cv.GaussianBlur(imgray, (7, 7), cv.BORDER_DEFAULT)
   
    _, threshold = cv.threshold(blur, threshold,255, cv.THRESH_BINARY)
    # Find Canny edges 
    edged = cv.Canny(threshold,minThreshold, maxThreshold) 



    contours, hierarchy = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cordinats = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea > area > minArea:
            
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if minPoints <= len(approx) <= maxPoints:
                cv.drawContours(imageToDrawOn, cnt, -1, (rgb_Color), 3) 
                x, y, w, h = cv.boundingRect(approx)
                center_x = x + w // 2
                center_y = y + h // 2
                x_cart, y_cart = ComputerVision.ImageProcessor.convert_to_cartesian((center_x, center_y))
                x_cart = round(x_cart, 2)
                y_cart = round(y_cart, 2)
                cv.putText(imageToDrawOn, f"{name}: {len(approx)}", (x + w + 20, y - 5), cv.FONT_HERSHEY_COMPLEX, 0.7, rgb_Color, 1)
                cv.putText(imageToDrawOn, f"Points: {len(approx)}", (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.7, rgb_Color, 1)
                cv.putText(imageToDrawOn, f"Area: {int(area)}", (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, 0.7, rgb_Color, 1)
                cv.putText(imageToDrawOn, f"(x,y): ({x_cart},{y_cart})", (x + w + 20, y + 70), cv.FONT_HERSHEY_COMPLEX, 0.7, rgb_Color, 1)
                cordinats.append((x_cart, y_cart))
                for point in approx:
                    cv.circle(imageToDrawOn, tuple(point[0]), 5, (rgb_Color), 1)  # Adjust circle radius

    return edged,imageToDrawOn,cordinats

def custom_object_detection(imageToDetectOn, imageToDrawOn):
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
    minPoints = cv.getTrackbarPos('minPoints', 'Trackbars')
    maxPoints = cv.getTrackbarPos('maxPoints', 'Trackbars')
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
            
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if minPoints <= len(approx) <= maxPoints:
                cv.drawContours(imageToDrawOn, cnt, -1, (0, 255, 0), 3) 
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
    
    
    return edged,imageToDrawOn