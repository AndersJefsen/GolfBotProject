from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
import path
import asyncio
import com
import numpy as np
import sys
import os
import detectionTools
import visualisation
from data import Data as Data
import imageManipulationTools
import time
import runFlow as rf
from queue import Queue
from time import time, strftime, gmtime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision



    

def høvl(data: Data,robot=True, image=None ):
        if(data.robot.detected and data.getAllBallCordinates() is not None):
                    currMidpoint,currAngle = data.robot.get_best_robot_position()
                    

                    contours=[]
                    egg_contour = data.egg.con
                    if egg_contour is not None:
                        contours.extend(egg_contour)
                        crosscon= data.cross.con
                    if crosscon is not None:
                        contours.extend(crosscon)
                    orange_ball_contour = data.orangeBall.con
                    if orange_ball_contour is not None:
                        contours.extend(orange_ball_contour)
                    
                    helpPoints=data.helpPoints
                    
                    drivepoints=data.drivepoints

                    closest_help_point, selected_ball,best_angle_to_turn, min_distance = path.find_shortest_path(data.robot.midpoint,data.robot.angle, helpPoints, contours,drivepoints) #data.helppoints.coords"""
                    #for point in helppoints:
                        #cv.circle(image, (int(point.con[0]), int(point.con[1])), 5, (0, 255, 0), -1)  # Green points
                    #if closest_help_point and selected_ball:
                        #cv.line(image, (int(currMidpoint[0]), int(currMidpoint[1])), (int(closest_help_point[0]), int(closest_help_point[1])), (0, 0, 255), 2)
                    if closest_help_point and selected_ball:
                        cv.line(image, (int(data.robot.midpoint[0]), int(data.robot.midpoint[1])), (int(closest_help_point[0]), int(closest_help_point[1])), (0, 0, 255), 2)  # Red line for movement
                    if(image is not None):
                # Resize the image
                        desired_size = (1200, 800)
                        resized_image = cv.resize(image, desired_size, interpolation=cv.INTER_LINEAR)
                        cv.imshow("Resized Image", resized_image)
                        cv.waitKey(1)
                    if robot:
                        com.drive_robot_to_point(ComputerVision.ImageProcessor.convert_to_cartesian(closest_help_point),ComputerVision.ImageProcessor.convert_to_cartesian(data.robot.midpoint),data.robot.angle,data.socket)
                        com.drive_robot_to_point(ComputerVision.ImageProcessor.convert_to_cartesian(selected_ball),ComputerVision.ImageProcessor.convert_to_cartesian(closest_help_point),data.robot.angle,data.socket)




def main(mode):
    global last_ball_detection_time
    data = Data()
    data.mode = mode
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
        video_path = "../master.mp4"  # Specify the path to the video file in the parent folder
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
    
    gui = False

    vision_image = Vision('ball.png')

    vision_image.init_control_gui()

    data.testpicturename = 'master.jpg'

    


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
                # Resize the image
                        desired_size = (1200, 800)
                        resized_image = cv.resize(data.output_image, desired_size, interpolation=cv.INTER_LINEAR)
                        cv.imshow("Resized Image", resized_image)
                        cv.waitKey(1)

        print("finding arena")
        try:
            findArena = rf.findArena_flow(data.screenshot,data.output_image,data)
        except Exception as e:
            print(f"An error occurredin rf.findArena_flow: {e}")
            break


  



    loop_time = time()
    
    while(True):
        try:
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
            #if data.cross.con is not None:
               # data.find_Cross_HP()
      
            data.find_Corner_HP()
            data.find_outer_ball_HP()
            
            rf.drawAndShow(data)

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
                        rf.update_positions(data,True,False,False,False,False,10)
                        newpos =  data.robot.get_best_robot_position()
                        if(newpos is not None):
                            currMidpoint,currAngle = newpos
                            correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(currMidpoint)
                            #turn to corrected position
                            rf.drawAndShow(data)
                            com.command_robot_turn(correctmid, data.getAllBallCordinates(),currAngle,data.socket)
                            print("command robot done")
                        #move
                        com.command_robot_move(correctmid, data.getAllBallCordinates(),data.socket)
                        print("command robot done")
                    else:
                        print("No best position found")
                    #start_time = time.time()

                    #last_ball_detection_time = time.time()
                    #loop_time = time.time()
                while len(data.whiteballs) == 0:
                    print("Operation Messi Commenced - wait ")
                            # Load the small goal
                    currMidpoint,currAngle = data.robot.get_best_robot_position()
                    correctmidCorrect = ComputerVision.ImageProcessor.convert_to_cartesian(
                    currMidpoint)

                    target_point = (150, 61)
                    result = com.move_to_position_and_release(target_point, correctmidCorrect, currAngle, data.socket)
                    if result:
                        print("Operation BigGOALGOAL successful")
                    else:
                        print("Operation Goal got fuckd mate")
                    break

            if(mode == "test"):
                høvl(data,False, data.output_image)
            if (mode== "videotest"):
                høvl(data,False, data.output_image)
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
        except Exception as e:
            print(f"An error occurred while trying to detect objects: {e}")
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