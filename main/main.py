from time import time
import cv2 as cv
from vision import Vision

import path
import com
import numpy as np
import sys
import os
import detectionTools
from data import Data as Data
import imageManipulationTools
import time
import runFlow as rf
from queue import Queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision

def main(mode):
    data = Data()
    data.mode = mode
    start_time = time.time()
    timer_duration = 420 #this is seconds so 6.5 minutes
    if mode == "camera" or mode == "robot" or mode == "Goal":
        data.wincap = cv.VideoCapture(0,cv.CAP_DSHOW)
        print("camera mode")
    elif mode == "window":
        from windowcapture import WindowCapture

        data.wincap = WindowCapture(None)
        print("window mode")
    elif mode == "test":
        print("test mode")
   
    elif mode == "videotest":
        video_path = "RobotVideo1.mp4"  # Specify the path to the video file in the parent folder
        data.wincap = cv.VideoCapture(video_path)
        if not data.wincap.isOpened():
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
    data.testpicturename = 'billede6(bold i hjørnet).png'
    findArena = False
    while not findArena:
        try:
            data.screenshot = None
            while(data.screenshot is None):
                data.screenshot=rf.getPicture(data)
                data.output_image = data.screenshot.copy()

                if data.screenshot is None:
                    print("Failed to capture screenshot.")
                    continue
        except Exception as e:
                print(f"An error occurred while trying to detect arena: {e}")
                break
        if(data.output_image is not None):
               rf.drawAndShow(data,"Resized Image")

        print("finding arena")
        try:
            findArena = rf.findArena_flow(data.screenshot,data.output_image,data)
        except Exception as e:
            print(f"An error occurredin rf.findArena_flow: {e}")
            break

    
    while(True):
    
        if(data.robot.detected):
            data.resetRobot()
        data.output_image = None
        #this is how many iterations you want to run detection on before sending the robot commands with the collected data
        #if set to 1 its working like it did before
        print("update positions")
        rf.update_positions(data,True,True,True,True,True,30)
        print("done updating positions")
        
        # painting time
        
        data.helpPoints = []
       
        print("finding helppoints")
        data.find_HP()
      
        
        print("Drawing image and showing it")
        rf.drawAndShow(data,"Resized Image")
        elapsed_time = time.time() - start_time
        remaining_time = timer_duration - elapsed_time
        print(f"Time remaining for mandetory messi: {remaining_time:.2f} seconds")

        if(mode == "robot" ):
            if(remaining_time < 0):
                print("Time is up")
                rf.messi(data)
            elif(remaining_time < 40):
                if(data.orangeHelpPoint is not None):
                    print("getting orangeBall")
                    rf.høvlOrange(data)
                    rf.messi(data)
                 
            elif(data.robot.detected and data.getAllBallCordinates()):
                data.timesNotDetected = 0
                data.robot.set_min_detections(5)
                print("HØVL MODE")
                rf.høvl(data,True, data.output_image)
                print("HØVL DONE")
            

            elif len(data.whiteballs) == 0:
                print("No white balls detected trying to get orangeball")
                        # Load the small goal'
                if(data.orangeHelpPoint is not None):
                    print("getting orangeBall")
                    rf.høvlOrange(data)
                else:
                    print("Operation Messi Commenced - wait ")
                    rf.messi(data)
                
        
        if(mode == "test"):
            rf.høvl(data,False, data.output_image)
        if (mode== "videotest"):
            rf.høvl(data,False, data.output_image)
        if (mode == "Goal"):
            if data.robot.detected:
                currMidpoint,currAngle = data.robot.get_best_robot_position()
                print("Robot orientation:")
                print(currAngle)
                print(data.robot.midpoint)
                correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(
                    currMidpoint
                )

                target_point = (12, 61.5)

                result = com.move_to_position_and_release(target_point, correctmid, currAngle, data.socket)
                if result:
                    print("Operation BigGOALGOAL successful")
                else:
                    print("Operation Goal got fuckd mate")

        if (mode == "camera"):

            image_center = (data.screenshot.shape[1] // 2, data.screenshot.shape[0] // 2)
            #print("Image Center - Pixel Coordinates:", image_center)
            cv.circle(data.screenshot,image_center, radius= 10 , color=(255,0,0),thickness=-1)

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
                closest_ball, distance_to_ball, angle_to_turn = path.find_close_ball(correctmid, data.getAllBallCordinates(), currAngle)
                print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")

                print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                


                    
    cv.destroyAllWindows()
    if mode == "window":
        data.wincap.release()
    if(data.socket != None):
        com.close_connection(data.socket)
    print('Done.')



if __name__ == "__main__":

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("No mode specified. Usage: python script_name.py <test|window|camera>")