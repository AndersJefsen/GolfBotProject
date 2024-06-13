from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
from path import find_close_ball
import asyncio
import com
import numpy as np
import sys
import os
import detectionTools
import imageManipulationTools
from queue import Queue
from time import time, strftime, gmtime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision 

def main(mode):
    if mode == "camera" or mode == "robot":  
        wincap = cv.VideoCapture(0,cv.CAP_DSHOW)
        print("camera mode")
    elif mode == "window":
        from windowcapture import WindowCapture

        wincap = WindowCapture(None)
        print("window mode")
    elif mode == "test":
        print("test mode")
    elif mode == "videotest":
        video_path = "../badvideo.mp4"  # Specify the path to the video file in the parent folder
        wincap = cv.VideoCapture(video_path)
        if not wincap.isOpened():
            print("Error: Could not open video file.")
            return
        print("videotest mode")    
    else:
        print("Invalid mode")
        return
    
    socket = None
    if mode == "robot":
        socket = com.connect_to_robot()


    arenaCorners = []
    mask = None
     
  
   
    gui = False
    
    vision_image = Vision('ball.png')
  
    vision_image.init_control_gui()
    
    testpicturename = 'testpic2.jpg'

    findArena = False
    
    while not findArena:
        try:
            if mode == "camera" or mode == "robot" or mode == "videotest":
                ret, screenshot = wincap.read()
            elif mode == "window":
                screenshot = wincap.get_screenshot()
            elif mode == "test":
                
                screenshot = cv.imread(testpicturename)
                
            output_image = screenshot.copy()
            
            if screenshot is None:
                print("Failed to capture screenshot.")
                continue
            print("finding arena")
            findArena, output_image,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ComputerVision.ImageProcessor.find_Arena(screenshot, output_image)
            print("her",findArena,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)
            if findArena:
                arenaCorners.append(bottom_left_corner)
                arenaCorners.append(bottom_right_corner)
                
                arenaCorners.append(top_right_corner)
                arenaCorners.append(top_left_corner)
                mask = imageManipulationTools.createMask(screenshot,arenaCorners)
                print("mask Created")
                break

        except Exception as e:
            print(f"An error occurred while trying to detect arena: {e}")
            break
    
            

    loop_time = time()
    while(True):
        try:
            if mode == "camera" or mode == "robot" or mode == "videotest":
                ret, screenshot = wincap.read()
            elif mode == "window":
                screenshot = wincap.get_screenshot()
            elif mode == "test":
                screenshot = cv.imread(testpicturename)
            if screenshot is None:
                print("Failed to capture screenshot.")
                continue
            
            ballcordinats = []
            eggcordinats = []
            orangecordinats = []
            robotcordinats = []
            crosscordinats = []
            inputimg = imageManipulationTools.useMask(screenshot,mask)
            #timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
            #cv.imwrite("test_"+timestamp+".jpg", screenshot)
            #inputimg = screenshot
            output_image = inputimg.copy()
            outputhsv_image = vision_image.apply_hsv_filter(inputimg)
            #cross
            edged, output_image,crosscordinats = detectionTools.detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 0, 0, 179, 255, 255, 0, 0, 0, 0), 100,200,1500,2000,"cross",(0, 255, 255),147,12,15,arenaCorners)
            #egg
            edged, output_image,eggcordinats = detectionTools.detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 0, 243, 179, 255, 255, 0, 0, 0, 0), minThreshold=100,maxThreshold=200,minArea=100,maxArea=600,name ="egg",rgb_Color=(255, 0, 204),threshold=227,minPoints=7,maxPoints=12,arenaCorners=arenaCorners)
            #orange
            edged, output_image,orangecordinats = detectionTools.detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 54, 0, 179, 255, 255, 0, 0, 0, 0), minThreshold=100,maxThreshold=200,minArea=50,maxArea=200,name ="orange",rgb_Color=(183, 102, 52),threshold=178,minPoints=6,maxPoints=10,arenaCorners=arenaCorners)
            #robot
            ballcontours = ComputerVision.ImageProcessor.find_balls_hsv(inputimg)
            if ballcontours is not None:
                ballcordinats, output_image = ComputerVision.ImageProcessor.convert_balls_to_cartesian(output_image, ballcontours)
            
            midpoint, angle, output_image = ComputerVision.ImageProcessor.process_robot(inputimg,output_image)

            cv.imshow("pic",output_image)

            if(mode == "robot" ):
                if(angle is not None and midpoint is not None and ballcordinats):
                    correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(midpoint)
                    
                    print("Robot orientation sss:")
                    print(angle)
                   
                    print("command robot")
                    com.command_robot(correctmid, ballcordinats, angle,socket)
                    print("command robot done")
            if(mode == "test"):
                  if(angle is not None and midpoint is not None and ballcordinats):
                        print("Robot orientation:")
                        print(angle)
                        correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(midpoint)
                        closest_ball, distance_to_ball, angle_to_turn = find_close_ball(correctmid, ballcordinats, angle)
                        print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")
    
                        print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
           
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"An error occurred while trying to detect objects: {e}")
            break

    cv.destroyAllWindows()
    if mode == "window":
        wincap.release()
    if(socket != None):
        com.close_connection(socket)
    print('Done.')



if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(main(sys.argv[1]))
    else:
        print("No mode specified. Usage: python script_name.py <test|window|camera>")