from data import Data as Data
import imageManipulationTools
import cv2 as cv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision
def findArena_flow(screenshot,output_image,data:Data):
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
                data.orangeBall.con = ComputerVision.ImageProcessor.find_orangeball_hsv(inputimg, 300, 1000)

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
                data.robot.con =ComputerVision.ImageProcessor.find_robot(inputimg, min_size=10, max_size=100000)
            
                angle = None
                img = data.screenshot

                if data.robot.con is not None:
                    if (len(data.robot.con)==3):
                        data.robot.originalMidtpoint, data.robot.angle, data.output_image, data.robot.direction=ComputerVision.ImageProcessor.getrobot(data.robot.con,data.output_image)
                        
                      #  data.robot.midpoint=ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0],data.robot.originalMidtpoint[1], data)
                        data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1], data)
                        data.robot.add_detection(data.robot.midpoint, data.robot.angle)
                    
                        data.robot.detected = True

                else:

                    print("Robot not detected in masked image, trying full image.")

                    data.robot.con = ComputerVision.ImageProcessor.find_robot(data.screenshot, min_size=10,
                                                                            max_size=100000)
      
                    if data.robot.con is not None and len(data.robot.con) == 3:
                              
                        data.robot.detected = True

                        data.robot.originalMidtpoint, data.robot.angle, data.output_image, data.robot.direction = ComputerVision.ImageProcessor.getrobot(
                            data.robot.con, data.output_image)

                        #data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(
                           # data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1],data)
                        data.robot.midpoint = ComputerVision.ImageProcessor.get_corrected_coordinates_robot(data.robot.originalMidtpoint[0], data.robot.originalMidtpoint[1], data)
                        data.robot.add_detection(data.robot.midpoint, data.robot.angle)

                    else:

                        data.robot.detected = False
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