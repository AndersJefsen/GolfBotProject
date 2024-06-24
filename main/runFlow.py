from data import Data as Data
import imageManipulationTools
import cv2 as cv
import numpy as np
import path
import com
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision
def findArena_flow(screenshot,output_image,data:Data):
    findArena, output_image,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, filtered_contoures = ComputerVision.ImageProcessor.find_Arena(screenshot, output_image)
    print("found arena corners",findArena,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)
    if findArena:
        arenaCorners = []
        arenaCorners.append(bottom_left_corner)
        arenaCorners.append(bottom_right_corner)

        arenaCorners.append(top_right_corner)
        arenaCorners.append(top_left_corner)
        data.arenaCorners = arenaCorners
        data.mask = imageManipulationTools.createMask(screenshot,arenaCorners)
        print("mask Created")
        
    return findArena
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

def getPicture(data: Data):
        mode = data.mode
        data.screenshot = None
        if mode == "camera" or mode == "robot" or mode == "Goal" or mode == "videotest":
            ret, data.screenshot = data.wincap.read()
        elif mode == "window":
            data.screenshot = data.wincap.get_screenshot()
        elif mode == "test":
            data.screenshot = cv.imread(data.testpicturename)
            '''
        if screenshot is not None:
            # Get the resolution of the image
            while True:
                height, width, channels = screenshot.shape
                print(f"Resolution: {width}x{height}")
                '''
        if data.screenshot is not None:
            #whight, wlength = vision_image.get_hight_and_length
           
            if  mode == "camera" or mode == "robot" or mode == "Goal":
                data.screenshot = resize_with_aspect_ratio(data.screenshot, 2048, 1024)
              
               
            else:
                data.screenshot = cv.resize(data.screenshot, (2048,1024), interpolation=cv.INTER_AREA)

        return data.screenshot

def update_positions(data :Data,robot:bool,balls:bool,egg:bool,orange:bool, cross:bool,iteration:int):
        for i in range(iteration):
            #print("iteration: ", i)
            
            data.screenshot = None
            while(data.screenshot is None):
                data.screenshot=getPicture(data)
                if data.screenshot is None:
                    print("Failed to capture screenshot.")
                    if data.mode == "videotest":
                        data.wincap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        print("Restarting video.")
                    continue


            inputimg = imageManipulationTools.useMask(data.screenshot,data.mask)
            
            data.output_image = inputimg.copy()


            if egg:
                data.egg.con = ComputerVision.ImageProcessor.find_bigball_hsv(inputimg, 2000, 8000)

            if orange:
                #data.orangeBall.con = None
                orangecon = ComputerVision.ImageProcessor.find_orangeball_hsv(inputimg)
                if orangecon is not None:
                    ''' print("DEBUG: ", len(orangecon))
                    for con in orangecon:
                            print("DEBUG Area: ", cv.contourArea(con))'''
                    if len(orangecon) > 1:
                        print("WARNING MORE THAN ONE ORANGE BALL DETECTED")
                    data.orangeBall.con = orangecon
              


            if balls:
                ballcontours = ComputerVision.ImageProcessor.find_balls_hsv1(inputimg)
                if ballcontours is not None:
                    ballcordinats, data.output_image = ComputerVision.ImageProcessor.process_and_convert_contours(data.output_image, ballcontours)
                    data.addBalls(ballcontours, ballcordinats)

            if cross:
                crosscon = ComputerVision.ImageProcessor.find_cross_contours(inputimg)
                if crosscon is not None:
                    data.cross.con = crosscon
                if  data.cross.con is not None:
                    data.cross.corner_con = ComputerVision.ImageProcessor.find_cross_corners(data.cross.con)


            if robot:
                
                data.robot.con =ComputerVision.ImageProcessor.find_robot(inputimg, min_size=200, max_size=1000)
            
                if data.robot.con is not None:
                    ComputerVision.ImageProcessor.show_contours_with_areas(inputimg, data.robot.con)
                #angle = None
                #img = data.screenshot
                if data.robot.con is not None:
                    if (len(data.robot.con)==3):
                        data.robot.originalMidtpoint, data.robot.angle, data.output_image, data.robot.direction=ComputerVision.ImageProcessor.getrobot(data.robot.con,data.output_image)
                        
                      #  data.robot.midpoint=ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0],data.robot.originalMidtpoint[1], data)
                        data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1], data)
                        #data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot_peter(data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1], data, data.arenaCorners)
                        data.robot.add_detection(data.robot.midpoint, data.robot.angle)
                    
                        data.robot.detected = True

                else:

                  

                    data.robot.con = ComputerVision.ImageProcessor.find_robot(data.screenshot, min_size=200,
                                                                            max_size=1000)
      
                    if data.robot.con is not None and len(data.robot.con) == 3:
                              
                        data.robot.detected = True

                        data.robot.originalMidtpoint, data.robot.angle, data.output_image, data.robot.direction = ComputerVision.ImageProcessor.getrobot(
                            data.robot.con, data.output_image)

                        #data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(
                           # data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1],data)
                        data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1], data)
                        #data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot_peter(data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1], data, data.arenaCorners)
                        data.robot.add_detection(data.robot.midpoint, data.robot.angle)

                    else:

                        data.robot.detected = False
        
        print("done with iterations in update position robot detected = ",data.robot.detected )
def paint_output(data: Data, output_image):
      #print("painting")
      #paint white balls
      output_image=ComputerVision.ImageProcessor.paintballs(data.getAllBallContours(), "ball", output_image)
                    #ComputerVision.ImageProcessor.showimage("balls", outputimage)
      #paint egg
      output_image=ComputerVision.ImageProcessor.paintballs(data.egg.con, "egg", output_image)
      #paint orang              #ComputerVision.ImageProcessor.showimage("egg", outputimage
      output_image=ComputerVision.ImageProcessor.paintballs(data.orangeBall.con, "orange", output_image)
                    #ComputerVision.ImageProcessor.showimage("final", outputimage)
      if data.cross.corner_con is not None:
            output_image = ComputerVision.ImageProcessor.draw_cross_corners(output_image, data.cross.corner_con)
      if data.orangeHelpPoint is not None:
      
        cords =  [np.array(contour, dtype=np.int32) for contour in data.orangeHelpPoint.con]
        pa = []
        
        pa.append(cords)
     
        imageManipulationTools.drawHelpPoints(output_image, pa)
      imageManipulationTools.drawHelpPoints(output_image, data.getAllHelpPointsCon())
     
      imageManipulationTools.drawHelpPoints(output_image, data.drivepoints,color=(0, 255, 0))
      #print("done painting")
      for area in data.outerArea.areas:
          index = 0
          if area.type == "BL_corner" or area.type == "BR_corner" or area.type == "TR_corner" or area.type == "TL_corner":
              color = (0, 255, 0)
             
          elif area.type == "cross_corner":
            if index == 0:
                color = (179, 0, 255)
            elif index == 1:
                color = (0, 255, 255)
            elif index == 2:
                color = (255, 0, 255)
            elif index == 3:
                color = (255, 255, 0)
            elif index == 4:
                color = (255, 0, 0)
            else:
                color = (0, 0, 0)
            index = index + 1
          else:
              color = (0, 0, 255)

          output_image = imageManipulationTools.drawArea(output_image, area.points,color)
      
        #Skal laves om
      #output_image = ComputerVision.ImageProcessor.draw_cross_corners(data.cross.con, output_image)
      
      if data.robot.detected:
        
        output_image=ComputerVision.ImageProcessor.paintballs(data.robot.con, "robo ball", output_image)
        output_image=ComputerVision.ImageProcessor.paintrobot(data.robot.originalMidtpoint, data.robot.angle, output_image, data.robot.direction)
        output_image=ComputerVision.ImageProcessor.paintrobot(data.robot.midpoint, data.robot.angle, output_image, data.robot.direction)
     
      return output_image

def drawAndShow(data:Data,windowName):
    data.output_image = paint_output(data, data.output_image)
    if(data.output_image is not None):
        # Resize the image
                desired_size = (800, 600)
                resized_image = cv.resize(data.output_image, desired_size, interpolation=cv.INTER_LINEAR)
                cv.imshow(windowName, resized_image)
                cv.waitKey(1)

def getRobotAngle(data:Data, selected_point):
    newpos = None
    data.resetRobot()
    while newpos is None:
        
        update_positions(data,True,False,False,False,False,10)
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
        '''
        print("selected point: ", ComputerVision.ImageProcessor.convert_to_cartesian(selected_point))  
        print("robot angle: ", data.robot.angle)
        print("robot midpoint: ", corrmid)
       
        print("isball: ",isBall)
        print("ismiddleball: ",isMiddleBall)
        '''
        print("angle to turn to ball: ", angle_to_turn)
        drawAndShow(data,"Resized Image")
        if angle_to_turn < 2 and angle_to_turn > -2: 
            print("correct angle achived: ",angle_to_turn)
            break
        com.turn_Robot( angle_to_turn,data.socket)
        print("done turning")
    
    if isMiddleBall is True and iteration == 0:
        distance_to_drive = distance_to_drive/2
    if isGoal is True:
        print("ITS A GOALPOINT SPECIAL DRIVE ")
        com.drive_Goal(distance_to_drive,data.socket)   
        return 
    elif(data.is_pos_in_corner(selected_point)):
        print("ITS A CORNERPOINT SPECIAL DRIVE ")
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
            print("PATH NOT CLEAR TRYING TO FIND DRIVEPOINT")
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
           

def messi(data:Data):
    if(data.robot.midpoint is not None):
        target_point = (130, 60)
        goal_point = (160,60)
        robotPos = data.robot.midpoint

        ispath = path.is_path_clear(robotPos,(target_point),add_all_obstacles(data,withOrange=False))  
        if(ispath):
            print("PATH CLEAR")
            angleCorrectionAndDrive(data,ComputerVision.ImageProcessor.convert_to_pixel(target_point),isBall=False,isMiddleBall=False)
            angleCorrectionAndDrive(data,ComputerVision.ImageProcessor.convert_to_pixel(goal_point),isBall=False,isMiddleBall=False,isGoal=True)
        else:
            print("PATH NOT CLEAR TRYING TO FIND DRIVEPOINT")
            drivepoint, drive_point_distance, drive_angle_to_turn = path.find_close_ball(robotPos, data.drivepoints, data.robot.angle)
            distance = path.calculate_distance_correct(robotPos, drivepoint)

            if distance < 50:
                index= data.drivepoints.index(drivepoint)
                newindex = (index+1)%4
                print("closest drivepoint: ", drivepoint)
                drivepoint = data.drivepoints[newindex]
            angleCorrectionAndDrive(data,drivepoint,isBall=False,isMiddleBall=False)


    
def høvl(data: Data,robot=True, image=None):
        if(data.robot.detected and data.getAllBallCordinates() is not None):
                    
                    #currMidpoint,currAngle = data.robot.get_best_robot_position()
                    contours = []
                    helpPoints = []
                    
                    contours= add_all_obstacles(data,withOrange=True)
                    helpPoints=data.helpPoints
                   
                    
                    
                    
                    
                    drivepoints=data.drivepoints
                    
                   
                    closest_help_point, selected_ball,best_angle_to_turn, min_distance = path.find_shortest_path(data.robot.midpoint,data.robot.angle, helpPoints, contours,drivepoints) #data.helppoints.coords"""
                  

                    
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
                           
                           
                            if distance > 5:
                                print("drive to first point")
                                angleCorrectionAndDrive(data,closest_help_point,isBall=False,isMiddleBall=False)
                                print("drive to second point")
                                angleCorrectionAndDrive(data,selected_ball,isBall=True,isMiddleBall=False)
                            else:
                                print("drive helpoint which is also helppoint")
                                angleCorrectionAndDrive(data,closest_help_point,isBall=True, isMiddleBall=True,iteration=0)
                    else:
                        print("No availerble helpoint trying drivepoint")
                        angleCorrectionAndDrive(data,closest_help_point,isBall=False,isMiddleBall=False)