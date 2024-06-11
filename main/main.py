from windowcapture import WindowCapture
from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
from com import command_robot
import numpy as np
import sys
import os

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
                x_cart, y_cart = ComputerVision.ImageProcessor.convert_to_cartesian((center_x, center_y), arenaCorners[0], arenaCorners[1], arenaCorners[3], arenaCorners[2])
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
def findWhiteBalls(imageToDetectOn, imageToDrawOn,vision_image):
       #this hsv filter is used to find the edges of the obstacles
    hsv_filter = HsvFilter(0, 0, 0, 179, 28, 255, 0, 0, 0, 0)
    img = vision_image.apply_hsv_filter(imageToDetectOn, hsv_filter)
    

    h, s, v = cv.split(img)
    #imgBlur = cv.GaussianBlur(imageToFindObstaclesIn, (7, 7), 1)
    #cv.imshow('blur', imgBlur)
    #so i proberly allready have the gray scale image

    
    imgray = v


    blur = cv.GaussianBlur(imgray, (7, 7), cv.BORDER_DEFAULT)
    thresholdmin= 161
    thresholdmax = 255
    _, threshold = cv.threshold(blur, thresholdmin,thresholdmax, cv.THRESH_BINARY)
    #rzb = cv.resize(threshold, (960, 540))
    #cv.imshow('thresh', rzb)
    
    #the strength of the edge detection where edxes between are considered weak edges and
    # are only shown if they are connected to strong edges which is above maxVal
  
    threshold1 = 0
    threshold2 = 200
    minArea = 50
    maxArea = 200

    # Find Canny edges 
    edged = cv.Canny(threshold,threshold1, threshold2) 



    contours, hierarchy = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea > area > minArea:
            
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if 6 <= len(approx) <= 10:
                cv.drawContours(imageToDrawOn, cnt, -1, (0, 0, 255), 3) 
                x, y, w, h = cv.boundingRect(approx)
                #cv.rectangle(imageToDrawOn, (x, y), (x + w, y + h), (252, 3, 3), 5)
                cv.putText(imageToDrawOn, "Balls: " + str(len(approx)), (x + w + 20, y + -5), cv.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)
                cv.putText(imageToDrawOn, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)
                cv.putText(imageToDrawOn, "Area: " + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)
                for point in approx:
                    cv.circle(imageToDrawOn, tuple(point[0]), 5, (255, 0, 0), 3)  # Adjust circle radius

    return edged,imageToDrawOn
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

def findArena(imageToDetectOn, imageToDrawOn): 
    hsv_filter = HsvFilter(0, 104, 0, 179, 255, 255, 0, 0, 0, 0)
    img = Vision.apply_hsv_filter(imageToDetectOn, hsv_filter)
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

def main(mode):
    if mode == "camera":  
        wincap = cv.VideoCapture(0,cv.CAP_DSHOW)
        print("camera mode")
    elif mode == "window":
        wincap = WindowCapture(None)
        print("window mode")
    elif mode == "test":
        print("test mode")
    else:
        print("Invalid mode")
        return
    
    arenaCorners = []
    mask = None
    

    vision_image = Vision('ball.png')

    vision_image.init_control_gui()

    findArena = False
    
    while not findArena:
        try:
            if mode == "camera":
                ret, screenshot = wincap.read()
            elif mode == "window":
                screenshot = wincap.get_screenshot()
            elif mode == "test":
                screenshot = cv.imread('testpic.jpg')
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
            print(f"An error occurred while trying to detect arena: {e}")
            break
    
            

    loop_time = time()
    while(True):
        try:
            if mode == "camera":
                ret, screenshot = wincap.read()
            elif mode == "window":
                screenshot = wincap.get_screenshot()
            elif mode == "test":
                screenshot = cv.imread('testpic.jpg')
            if screenshot is None:
                print("Failed to capture screenshot.")
                continue
            
            ballcordinats = []
            eggcordinats = []
            orangecordinats = []
            robotcordinats = []
            crosscordinats = []

            inputimg = useMask(screenshot,mask)
            #inputimg = screenshot
            output_image = inputimg.copy()
            #ballcon, output_image = ComputerVision.ImageProcessor.find_balls(inputimg,output_image)
            outputhsv_image = vision_image.apply_hsv_filter(inputimg)
            #outputedge_image = vision_image.apply_edge_filter(outputhsv_image)
            edged, output_image,crosscordinats = detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 0, 0, 179, 255, 255, 124, 0, 0, 0), 1000,1000,400,800,"cross",(0, 255, 255),182,8,12,arenaCorners)
            #egg
            edged, output_image,eggcordinats = detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 0, 243, 179, 255, 255, 0, 0, 0, 0), minThreshold=100,maxThreshold=200,minArea=100,maxArea=600,name ="egg",rgb_Color=(255, 0, 204),threshold=227,minPoints=7,maxPoints=12,arenaCorners=arenaCorners)
            #orange
            edged, output_image,orangecordinats = detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 54, 0, 179, 255, 255, 0, 0, 0, 0), minThreshold=100,maxThreshold=200,minArea=50,maxArea=200,name ="orange",rgb_Color=(183, 102, 52),threshold=178,minPoints=6,maxPoints=10,arenaCorners=arenaCorners)
            #robot
            #edged, output_image = detect_objects(inputimg,output_image,vision_image, HsvFilter(91, 107, 0, 154, 255, 207, 0, 0, 0, 0), minThreshold=0,maxThreshold=1000,minArea=30,maxArea=200,name ="robot",rgb_Color=(0, 42, 255),threshold=28,minPoints=8,maxPoints=8)
            
            #edged, output_image = findWhiteBalls(inputimg,output_image,vision_image)
            edged, output_image,ballcordinats = detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 0, 0, 179, 28, 255, 0, 0, 0, 0), minThreshold=0,maxThreshold=200,minArea=50,maxArea=200,name ="ball",rgb_Color=(0, 0, 255),threshold=161,minPoints=6,maxPoints=10,arenaCorners=arenaCorners)

            ballcon, output_image,angle, midpoint = ComputerVision.ImageProcessor.find_robot_withOutput(inputimg,output_image,bottom_left_corner=arenaCorners[0], bottom_right_corner=arenaCorners[1], top_left_corner=arenaCorners[3], top_right_corner=arenaCorners[2])
            
            
            '''
            if(angle is not None and midpoint is not None):
             correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(midpoint, arenaCorners[0], arenaCorners[1], arenaCorners[3], arenaCorners[2])
             print(correctmid)
             screenoutput = cv.resize(output_image, (960, 540))
             cv.imshow('Computer Vision', screenoutput)
             cv.waitKey(0)
             command_robot(correctmid, ballcordinats, angle)
             break
            '''
            # edged, output_image = findRoundObjects(outputhsv_image,output_image)
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
            
            #rzhsv = cv.resize(outputhsv_image, (960, 540))
            #cv.imshow('hsv', rzhsv)
            screenoutput = cv.resize(output_image, (960, 540))
            cv.imshow('Computer Vision', screenoutput)
            rze = cv.resize(edged, (960, 540))
            cv.imshow('edges', rze)
            
        
            

            #screengray = cv.resize(imgray, (960, 540))
            
            #screenhsvimg = cv.resize(outputhsv_image, (960, 540))
            #screenedgeimg = cv.resize(outputedge_image, (960, 540))
            #cv.imshow('gray', screengray)
            #cv.imshow('hsv', screenhsvimg)
            #cv.imshow('edge', screenedgeimg)

        

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"An error occurred while trying to detect objects: {e}")
            break

    cv.destroyAllWindows()
    wincap.release()
    print('Done.')



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("No mode specified. Usage: python script_name.py <test|window|camera>")