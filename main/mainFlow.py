import cv2 as cv
from data import Data as Data
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

def getPicture(data:Data):
        mode = data.mode
        wincap = data.wincap
        if mode == "camera" or mode == "robot" or mode == "Goal" or mode == "videotest":
            ret, data.screenshot = wincap.read()
        elif mode == "window":
            data.screenshot = wincap.get_screenshot()
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

def update_positions(data:Data,robot:bool,balls:bool,egg:bool,orange:bool, cross:bool,iteration:int):
            
            for i in range(iteration):
                    print("iteration: ", i)
                    #print("iteration: ", i)
                    
                    data.screenshot = None
                    while(data.screenshot is None):
                        data.screenshot=getPicture()
                        if data.screenshot is None:
                            print("Failed to capture screenshot.")
                            if data.mode == "videotest":
                                data.wincap.set(cv.CAP_PROP_POS_FRAMES, 0)
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
            return output_image