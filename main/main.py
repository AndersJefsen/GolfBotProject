from time import time
import cv2 as cv
from vision import Vision

import path
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision

def getRobotAngle(data:Data, selected_point):
    newpos = None
    data.resetRobot()
    while newpos is None:
        
        rf.update_positions(data,True,False,False,False,False,10)
        newpos =  data.robot.get_best_robot_position()
        if(newpos is not None):
            data.timesNotDetected =0
            currMidpoint,currAngle = newpos
            correctmid = ComputerVision.ImageProcessor.convert_to_cartesian(currMidpoint)
            angle_to_turn = path.calculate_angle(correctmid,ComputerVision.ImageProcessor.convert_to_cartesian(selected_point),currAngle)
            distance_to_drive = path.calculate_distance(correctmid,ComputerVision.ImageProcessor.convert_to_cartesian(selected_point))
        else:
            data.timesNotDetected += 1
            if data.timesNotDetected >=10:
                com.turn_Robot(50,data.socket)       
                data.timesNotDetected = 0
                            
    return angle_to_turn,distance_to_drive, correctmid
     

def angleCorrectionAndDrive(data:Data, selected_point, isBall = False,isMiddleBall = False,iteration = 1,isGoal = False):
  
    while True:
        angle_to_turn, distance_to_drive, corrmid =  getRobotAngle(data,selected_point)
        print("selected point: ", ComputerVision.ImageProcessor.convert_to_cartesian(selected_point))  
        print("robot angle: ", data.robot.angle)
        print("robot midpoint: ", corrmid)
        print("angle to turn to ball: ", angle_to_turn)
        print("isball: ",isBall)
        print("ismiddleball: ",isMiddleBall)

        
        rf.drawAndShow(data,"Resized Image")
        if angle_to_turn < 2 and angle_to_turn > -2: 
            print("correct angle achived: ",angle_to_turn)
            break
        com.turn_Robot( angle_to_turn,data.socket)
        print("done turning")
    
    if isMiddleBall is True and iteration == 0:
        distance_to_drive = distance_to_drive/2
    if isGoal is True:
        print("ITS A GOALPOINT SPECIAL DRIVE sleeping")
        com.drive_Goal(distance_to_drive,data.socket)   
        return 
    elif(data.is_pos_in_corner(selected_point)):
        print("ITS A CORNERPOINT SPECIAL DRIVE sleeping")
        com.drive_Robot_Corner(distance_to_drive,data.socket)
    else:
        com.drive_Robot(distance_to_drive,data.socket)
    print("done driving")

    if isBall is False or iteration == 0:
        angle_to_turn, distance_to_drive, corrmid =  getRobotAngle(data,selected_point)
        print("check angle: ", angle_to_turn)
        print("check distance: ", distance_to_drive)

        if distance_to_drive > 10 or iteration ==0:
            print("distance to drive is: ", distance_to_drive)
         
            iteration = 1
            print("not arrived at point trying again with distance: ",distance_to_drive)
        
            angleCorrectionAndDrive(data,selected_point,isBall,isMiddleBall=False,iteration=iteration)
        else:
            print("arrived at point with distance: ",distance_to_drive)
      

def add_all_obstacles(data:Data, withOrange = True):
    contours = []
    egg_contour = data.egg.con
    if egg_contour is not None:
        contours.extend(egg_contour)
        crosscon= data.cross.con
    if crosscon is not None:

    
        contours.extend(crosscon)
    if withOrange:
        orange_ball_contour = data.orangeBall.con
        if orange_ball_contour is not None:
            contours.extend(orange_ball_contour)    
    return contours

def høvlOrange(data:Data):
    print("HØVL ORANGE")
   

    points = []
    isDrivePoint = False
    if(data.orangeHelpPoint is not None and data.robot.midpoint is not None):
        robotPos = data.robot.midpoint
        hp_x = data.orangeHelpPoint.con[0]
        hp_y = data.orangeHelpPoint.con[1]
       
        

        ispath = path.is_path_clear(robotPos,(hp_x,hp_y),add_all_obstacles(data,withOrange=False))
        if(ispath): 
            print("PATH CLEAR")
          
            
            for i, contour in enumerate(data.orangeBall.con, 1):
                x, y, w, h = cv.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
            if center_x == hp_y and center_y == hp_y:
                print("orange ball is at help point")
                points.append((center_x,center_y))
            else:
                print("orange ball is not at help point")
                points.append((center_x,center_y))
                points.append((center_x,center_y))
                
            
        else:
            drivepoint, drive_point_distance, drive_angle_to_turn = path.find_close_ball(robotPos, data.drivepoints, data.robot.angle)
            distance = path.calculate_distance_correct(robotPos, drivepoint)

            if distance < 50:
                index= data.drivepoints.index(drivepoint)
                newindex = (index+1)%4
                print("closest drivepoint: ", drivepoint)
                drivepoint = data.drivepoints[newindex]
            points.append(drivepoint)
            isDrivePoint = True
        
        if len(points) == 1:
            if isDrivePoint:
                angleCorrectionAndDrive(data,points[0],isBall=False,isMiddleBall=False)
            else:
                 angleCorrectionAndDrive(data,points[0],isBall=True,isMiddleBall=True,iteration=0)
        else:
            angleCorrectionAndDrive(data,points[0],isBall=False,isMiddleBall=False)
            angleCorrectionAndDrive(data,points[1],isBall=True,isMiddleBall=False)
           

    
    
    
def høvl(data: Data,robot=True, image=None):
        if(data.robot.detected and data.getAllBallCordinates() is not None):
                    
                    currMidpoint,currAngle = data.robot.get_best_robot_position()
                    contours = []
                    helpPoints = []
                    
                    contours= add_all_obstacles(data,withOrange=True)
                    helpPoints=data.helpPoints
                   
                    
                    
                    
                    
                    drivepoints=data.drivepoints
                    print("drivepoints: ",drivepoints)
                   
                    closest_help_point, selected_ball,best_angle_to_turn, min_distance = path.find_shortest_path(data.robot.midpoint,data.robot.angle, helpPoints, contours,drivepoints) #data.helppoints.coords"""
                  

                    print("here")
                    if selected_ball is not None:
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
                            cp = np.array([closest_help_point[0],closest_help_point[1]])
                            bp = np.array([selected_ball[0],selected_ball[1]])
                            distance = np.linalg.norm(cp - bp)
                            print("distance between helppoint and ball: ",distance)
                           
                            if distance > 5:
                                print("drive to first point")
                                angleCorrectionAndDrive(data,closest_help_point,isBall=False,isMiddleBall=False)
                                print("drive to second point")
                                angleCorrectionAndDrive(data,selected_ball,isBall=True,isMiddleBall=False)
                            else:
                                print("drive helpoint which is also helppoint")
                                angleCorrectionAndDrive(data,closest_help_point,isBall=True, isMiddleBall=True,iteration=0)
                    else:
                        angleCorrectionAndDrive(data,closest_help_point,isBall=False,isMiddleBall=False)




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
        #if data.cross.con is not None:
            # data.find_Cross_HP()
        print("beforeHP")
        data.find_HP()
        print("AfterHP")
        
        print("before drawandshow")
        rf.drawAndShow(data,"Resized Image")
        print("after drawandshow")

        if(mode == "robot" ):
            if(data.robot.detected and data.getAllBallCordinates()):
                data.timesNotDetected = 0
                data.robot.set_min_detections(5)
                print("before høvl")
                høvl(data,True, data.output_image)
                print("after høvl")
            

            if len(data.whiteballs) == 0:
                print("No white balls detected trying to get orangeball")
                        # Load the small goal'
                if(data.orangeHelpPoint is not None):
                    print("getting orangeBall")
                    høvlOrange(data)
                else:
                    print("Operation Messi Commenced - wait ")
                    target_point = (130, 55)
                    goal_point = (160,55)
                    angleCorrectionAndDrive(data,ComputerVision.ImageProcessor.convert_to_pixel(target_point),isBall=False,isMiddleBall=False)
                    angleCorrectionAndDrive(data,ComputerVision.ImageProcessor.convert_to_pixel(goal_point),isBall=False,isMiddleBall=False,isGoal=True)
               
                
        
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