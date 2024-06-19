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
from data import Data as Data
import imageManipulationTools
from queue import Queue
from time import time, strftime, gmtime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision
def paint_output(data, output_image):
      #paint white balls
      output_image=ComputerVision.ImageProcessor.paintballs(data.getAllBallContours(), "ball", output_image)
                    #ComputerVision.ImageProcessor.showimage("balls", outputimage)
      #paint egg
      output_image=ComputerVision.ImageProcessor.paintballs(data.egg.con, "egg", output_image)
      #paint orang              #ComputerVision.ImageProcessor.showimage("egg", outputimage
      output_image=ComputerVision.ImageProcessor.paintballs(data.orangeBall.con, "orange", output_image)
                    #ComputerVision.ImageProcessor.showimage("final", outputimage)

        #Skal laves om
      #output_image = ComputerVision.ImageProcessor.draw_cross_corners(data.cross.con, output_image)

      if data.robot.detected:
        
        output_image=ComputerVision.ImageProcessor.paintballs(data.robot.con, "robo ball", output_image)
        output_image=ComputerVision.ImageProcessor.paintrobot(data.robot.originalMidtpoint, data.robot.angle, output_image, data.robot.direction)
        output_image=ComputerVision.ImageProcessor.paintrobot(data.robot.midpoint, data.robot.angle, output_image, data.robot.direction)

      return output_image
    
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
    data = Data() 
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
        video_path = 'Robot Video.mp4'  # Specify the path to the video file in the parent folder
        wincap = cv.VideoCapture(video_path)
        if not wincap.isOpened():
            print("Error: Could not open video file.")
            return
        print("videotest mode")
    else:
        print("Invalid mode")
        return

  
    if mode == "robot":
        data.socket = com.connect_to_robot()

    if mode == "Goal":
        data.socket = com.connect_to_robot()
    
    gui = False

    vision_image = Vision('ball.png')

    vision_image.init_control_gui()

    testpicturename = 'Kryds 1.jpg'

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
            screenshot = None
            while(screenshot is None):
                screenshot=getPicture()
                output_image = screenshot.copy()

                if screenshot is None:
                    print("Failed to capture screenshot.")
                    continue

            print("finding arena")
            findArena, output_image,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, filtered_contoures = ComputerVision.ImageProcessor.find_Arena(screenshot, output_image)
            print("her",findArena,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)
            if findArena:
                arenaCorners = []
                arenaCorners.append(bottom_left_corner)
                arenaCorners.append(bottom_right_corner)

                arenaCorners.append(top_right_corner)
                arenaCorners.append(top_left_corner)
                data.arenaCorners = arenaCorners
                data.mask = imageManipulationTools.createMask(screenshot,arenaCorners)
                print("mask Created")
                break

        except Exception as e:
            print(f"An error occurred while trying to detect arena: {e}")
            break


    def update_positions(robot:bool,balls:bool,egg:bool,orange:bool, cross:bool,iteration:int):
              for i in range(iteration):
                    #print("iteration: ", i)
                    
                    screenshot = None
                    while(screenshot is None):
                        screenshot=getPicture()
                        if screenshot is None:
                            print("Failed to capture screenshot.")
                            if mode == "videotest":
                                wincap.set(cv.CAP_PROP_POS_FRAMES, 0)
                                print("Restarting video.")
                            continue


                    inputimg = imageManipulationTools.useMask(screenshot,data.mask)
                   
                    output_image = inputimg.copy()
                    outputhsv_image = vision_image.apply_hsv_filter(inputimg)


                    if egg:
                        data.egg.con = ComputerVision.ImageProcessor.find_bigball_hsv(inputimg, 2000, 8000)

                    if orange:
                        data.orangeBall.con = ComputerVision.ImageProcessor.find_orangeball_hsv(inputimg, 300, 1000)

                    if balls:
                        ballcontours = ComputerVision.ImageProcessor.find_balls_hsv1(inputimg)
                        if ballcontours is not None:
                            ballcordinats, output_image = ComputerVision.ImageProcessor.process_and_convert_contours(output_image, ballcontours)
                            data.addBalls(ballcontours, ballcordinats)

                    if cross:
                        cross_contour = ComputerVision.ImageProcessor.find_cross_contours(inputimg)


                    if robot:
                        data.robot.con =ComputerVision.ImageProcessor.find_robot(inputimg, min_size=60, max_size=100000)
                        angle = None
                        img = screenshot

                        if data.robot.con is not None:
                            if (len(data.robot.con)==3):
                                data.robot.originalMidtpoint, data.robot.angle, output_image, data.robot.direction=ComputerVision.ImageProcessor.getrobot(data.robot.con,output_image)
                                
                                data.robot.midpoint=ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0],data.robot.originalMidtpoint[1])
                                
                                data.robot.add_detection(data.robot.midpoint, data.robot.angle)
                            
                                data.robot.detected = True

                        else:

                            print("Robot not detected in masked image, trying full image.")

                            data.robot.con = ComputerVision.ImageProcessor.find_robot(screenshot, min_size=60,
                                                                                    max_size=100000)

                            if data.robot.con is not None and len(data.robot.con) == 3:

                                data.robot.detected = True

                                data.robot.originalMidtpoint, data.robot.angle, output_image, data.robot.direction = ComputerVision.ImageProcessor.getrobot(
                                    data.robot.con, output_image)

                                data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(
                                    data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1])

                                data.robot.add_detection(data.robot.midpoint, data.robot.angle)

                            else:

                                data.robot.detected = False

                            

    loop_time = time()
    
    while(True):
        try:
            if(data.robot.detected):
                data.resetRobot()
            update_positions(True,True,True,True,True,30)


                    #ComputerVision.ImageProcessor.showimage("", outputimage)

                    #cross
                    #needs to be fixed
                    #data.cross.con, output_image_with_cross = ComputerVision.ImageProcessor.find_cross_contours( filtered_contoures, output_image)
                    #data.cross.cord, output_image_with_cross = ComputerVision.ImageProcessor.convert_cross_to_cartesian(data.cross.con, output_image_with_cross)

                  
           
            # painting time
            output_image = paint_output(data, output_image)
           

            if(output_image is not None):
                # Resize the image
                desired_size = (1200, 800)
                resized_image = cv.resize(output_image, desired_size, interpolation=cv.INTER_LINEAR)
                cv.imshow("Resized Image", resized_image)
                cv.waitKey(1)
               
            

            if(mode == "robot" ):
                if(data.robot.detected and data.getAllBallCordinates()):
                    data.robot.set_min_detections(10)
                    bestpos =  data.robot.get_best_robot_position()
                    if(bestpos is not None):
                        currMidpoint,currAngle = bestpos 
                        correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(currMidpoint)

                        print("Robot orientation sss:")
                        print(currAngle)

                        #first turn
                        print("command robot")
                        com.command_robot_turn(correctmid, data.getAllBallCordinates(),currAngle,data.socket)
                        print("command robot done")
                        
                        data.resetRobot()
                        #get new position
                        update_positions(True,False,False,False,False,10)
                        newpos =  data.robot.get_best_robot_position()
                        if(newpos is not None):
                            currMidpoint,currAngle = newpos
                            correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(currMidpoint)
                            #turn to corrected position
                            com.command_robot_turn(correctmid, data.getAllBallCordinates(),currAngle,data.socket)
                            print("command robot done")
                        #move
                        com.command_robot_move(correctmid, data.getAllBallCordinates(),data.socket)
                        print("command robot done")
                    else:
                        print("No best position found")
            if(mode == "test"):
           
                if(data.robot.detected and data.getAllBallCordinates()):
                    currMidpoint,currAngle = data.robot.get_best_robot_position()
                   # print("Robot orientation:")
                   # print(angle)
                    correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(currMidpoint)
                    closest_ball, distance_to_ball, angle_to_turn = find_close_ball(correctmid,data.getAllBallCordinates(), currAngle)
                   # print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")

                    #print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")

            if (mode == "Goal"):
                if data.robot.detected:
                    currMidpoint,currAngle = data.robot.get_best_robot_position()
                    print("Robot orientation:")
                    print(angle)
                    print(data.robot.midpoint)
                    correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(
                        currMidpoint, data.arenaCorners[0], data.arenaCorners[1], data.arenaCorners[3], data.arenaCorners[2]
                    )

                    target_point = (12, 61.5)

                    result = com.move_to_position_and_release(target_point, correctmid, currAngle, data.socket)
                    if result:
                        print("Operation BigGOALGOAL successful")
                    else:
                        print("Operation Goal got fuckd mate")

            if (mode == "camera"):

                image_center = (screenshot.shape[1] // 2, screenshot.shape[0] // 2)
                #print("Image Center - Pixel Coordinates:", image_center)
                cv.circle(screenshot,image_center, radius= 10 , color=(255,0,0),thickness=-1)

                image_center = ComputerVision.ImageProcessor.convert_to_cartesian(image_center)
                #print("Image Center - Cartisan coord q:", image_center)
                if data.robot.detected:
                     
                    currMidpoint,currAngle = data.robot.get_best_robot_position()
                    print(currAngle)
                    correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(
                        currMidpoint
                    )
                    print(correctmid)
                    #Ik ud med riven Viktor, det her for hjælp til Anders Offset beregning via fysisk pixel midterpunkt ud fra kameraet's position.
                    closest_ball, distance_to_ball, angle_to_turn = find_close_ball(correctmid, data.getAllBallCordinates(), currAngle)
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
    if(data.socket != None):
        com.close_connection(data.socket)
    print('Done.')

    


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("No mode specified. Usage: python script_name.py <test|window|camera>")