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
import visualisation
import imageManipulationTools
from queue import Queue
from time import time, strftime, gmtime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision

def resize_with_aspect_ratio(image, target_width, target_height):
    original_height, original_width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    
    # Calculate the scaling factors
    width_factor = target_width / original_width
    height_factor = target_height / original_height
    
    # Use the smaller scaling factor to keep aspect ratio
    scaling_factor = min(width_factor, height_factor)
    
    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    
    # Resize the image
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    return resized_image
def main(mode):
    if mode == "camera" or mode == "robot" or mode == "Goal":
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

    if mode == "Goal":
        socket = com.connect_to_robot()


    arenaCorners = []
    mask = None



    gui = False

    vision_image = Vision('ball.png')

    vision_image.init_control_gui()

    testpicturename = 'v.jpg'

    def getPicture():
        if mode == "camera" or mode == "robot" or mode == "Goal" or mode == "videotest":
            ret, screenshot = wincap.read()
        elif mode == "window":
            screenshot = wincap.get_screenshot()
        elif mode == "test":
            screenshot = cv.imread(testpicturename)
            '''
        if screenshot is not None:
            # Get the resolution of the image
            while True:
                height, width, channels = screenshot.shape
                print(f"Resolution: {width}x{height}")
                '''
        if screenshot is not None:
            #whight, wlength = vision_image.get_hight_and_length
           
            if  mode == "camera" or mode == "robot" or mode == "Goal":
                screenshot = resize_with_aspect_ratio(screenshot, 2048, 1024)
              
               
            else:
                screenshot = cv.resize(screenshot, (2048,1024), interpolation=cv.INTER_AREA)

        return screenshot


    findArena = False

    while not findArena:
        try:

            screenshot=getPicture()
            output_image = screenshot.copy()

            if screenshot is None:
                print("Failed to capture screenshot.")
                continue

            print("finding arena")
            findArena, output_image,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, filtered_contoures = ComputerVision.ImageProcessor.find_Arena(screenshot, output_image)
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
            screenshot=getPicture()
            if screenshot is None:
                print("Failed to capture screenshot.")
                if mode == "videotest":
                    wincap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    print("Restarting video.")
                continue


            inputimg = imageManipulationTools.useMask(screenshot,mask)
            #timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
            #cv.imwrite("test_"+timestamp+".jpg", screenshot)
            #inputimg = screenshot
            output_image = inputimg.copy()
            outputhsv_image = vision_image.apply_hsv_filter(inputimg)



            #egg
            #edged, output_image,eggcordinats = detectionTools.detect_objects(inputimg,output_image,vision_image, HsvFilter(0, 0, 243, 179, 255, 255, 0, 0, 0, 0), minThreshold=100,maxThreshold=200,minArea=100,maxArea=600,name ="egg",rgb_Color=(255, 0, 204),threshold=227,minPoints=7,maxPoints=12,arenaCorners=arenaCorners)
            eggcordinats = ComputerVision.ImageProcessor.find_bigball_hsv(inputimg, 2000, 8000)
            #orange
            orangecordinats = ComputerVision.ImageProcessor.find_orangeball_hsv(inputimg, 300, 1000)
            #balls
            ballcontours = ComputerVision.ImageProcessor.find_balls_hsv1(inputimg)
          

            if ballcontours is not None:
                #print("")
                ballcordinats, output_image = ComputerVision.ImageProcessor.process_and_convert_contours(output_image, ballcontours)


            robotcordinats=ComputerVision.ImageProcessor.find_robot(inputimg, min_size=0, max_size=100000)

            angle = None
            if robotcordinats is not None:
                if (len(robotcordinats)==3):
                    midpoint, angle, output_image, direction=ComputerVision.ImageProcessor.getrobot(robotcordinats,output_image)
                    output_image=ComputerVision.ImageProcessor.paintrobot(midpoint, angle, output_image, direction)
                    output_image=ComputerVision.ImageProcessor.paintballs(robotcordinats, "robo ball", output_image)
                    #print(midpoint)
                    midpoint=ComputerVision.ImageProcessor.get_corrected_coordinates_robot(midpoint[0],midpoint[1])
                    #print(midpoint)

                    output_image=ComputerVision.ImageProcessor.paintrobot(midpoint, angle, output_image, direction)



            #ComputerVision.ImageProcessor.showimage("", outputimage)

            #cross
            cross_counters, output_image_with_cross = ComputerVision.ImageProcessor.find_cross_contours( filtered_contoures, output_image)
            cartesian_cross_list, output_image_with_cross = ComputerVision.ImageProcessor.convert_cross_to_cartesian(cross_counters, output_image_with_cross)

            output_image=ComputerVision.ImageProcessor.paintballs(ballcontours, "ball", output_image_with_cross)
            #ComputerVision.ImageProcessor.showimage("balls", outputimage)

            output_image=ComputerVision.ImageProcessor.paintballs(eggcordinats, "egg", output_image)
            #ComputerVision.ImageProcessor.showimage("egg", outputimage)

            output_image=ComputerVision.ImageProcessor.paintballs(orangecordinats, "orange", output_image)
            #ComputerVision.ImageProcessor.showimage("final", outputimage)
            if(output_image is not None):
                # Resize the image
                desired_size = (1200, 800)
                resized_image = cv.resize(output_image, desired_size, interpolation=cv.INTER_LINEAR)
                cv.imshow("Resized Image", resized_image)
              

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

            if (mode == "Goal"):
                if angle is not None and midpoint is not None:
                    print("Robot orientation:")
                    print(angle)
                    print(midpoint)
                    correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(
                        midpoint, arenaCorners[0], arenaCorners[1], arenaCorners[3], arenaCorners[2]
                    )

                    target_point = (12, 61.5)

                    result = com.move_to_position_and_release(target_point, correctmid, angle, socket)
                    if result:
                        print("Operation BigGOALGOAL successful")
                    else:
                        print("Operation Goal got fuckd mate")

            if (mode == "camera"):

                image_center = (screenshot.shape[1] // 2, screenshot.shape[0] // 2)
                print("Image Center - Pixel Coordinates:", image_center)
                cv.circle(screenshot,image_center, radius= 10 , color=(255,0,0),thickness=-1)

                image_center = ComputerVision.ImageProcessor.convert_to_cartesian(image_center)
                print("Image Center - Cartisan coord q:", image_center)
                if angle is not None and midpoint is not None:

                    print(angle)
                    correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(
                        midpoint
                    )
                    print(correctmid)
                    #Ik ud med riven Viktor, det her for hjælp til Anders Offset beregning via fysisk pixel midterpunkt ud fra kameraet's position.


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
        main(sys.argv[1])
    else:
        print("No mode specified. Usage: python script_name.py <test|window|camera>")