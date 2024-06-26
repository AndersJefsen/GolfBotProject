#computer vision

"""
    #find egg
    @staticmethod
    def find_bigball_threshold(image, min_size=300, max_size=1000000000, threshold=180):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the grayscale image
        _, thresh_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

        # Display the processed image
        cv2.imshow('Processed Image', thresh_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_ball_contour = None

        # Logic to find the biggest ball
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.0001 <= circularity <= 500:
                    big_ball_contour = cnt
                    max_area = area

        return big_ball_contour

#find white balls
    @staticmethod
    def find_balls_threshold(image, min_size=300, max_size=1000000000, threshold=180):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the grayscale image
        _, thresh_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

        # Display the processed image
        cv2.imshow('Processed Image', thresh_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []

        # Logic to find contours of the balls
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    ball_contours.append(cnt)

        return ball_contours

 
    @staticmethod
    def find_robot_withOutput(inputImage,outPutImage, min_size=100, max_size=100000,bottom_left_corner=None, bottom_right_corner=None, top_left_corner=None, top_right_corner=None):
        hsv_image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([78,100,100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")

        # Threshhold the HSV image image to get only blue colors
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
       
        # Load maskerne på billedet

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet

        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []
        # Logikken for at finde countours på boldene
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    ball_contours.append(cnt)

        # Converter til cartesian coordinates
        roboball = ball_contours 
        
        for i, contour in enumerate(roboball, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(outPutImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(outPutImage, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                #print(f"roboBall {i} Cartesian Coordinates: {cartesian_coords}")
                
        
        midpoint, direction = ImageProcessor.find_direction(roboball)
        if roboball and len(roboball) == 3:
            if midpoint and direction:
                    # Draw the direction from the midpoint
                endpoint = (midpoint[0] + direction[0], midpoint[1] + direction[1])
                cv2.circle(outPutImage, midpoint, 10, (0, 0, 255), -1)  # Red dot at midpoint
                cv2.line(outPutImage, midpoint, endpoint, (255, 0, 0), 3)  # Blue line indicating direction
                #print("Midpoint:", midpoint)
                #print("Direction to third point:", direction)
                
        #else:
            #print("Could not find exactly three balls., found ", len(roboball))        
        

        angle = None
        if direction:
            angle = ImageProcessor.calculate_angle(direction)
        
        if angle is not None:
            print(f"Angle: {angle} degrees")
        print("here")
        return ball_contours, outPutImage, angle, midpoint
        
        """

'''
convert to cartisian 

 @staticmethod
    def convert_orangeball_to_cartesian(image, orangeball_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()

        output_Image = image.copy()
        cartesian_coords = None

        if orangeball_contours:
            contour = orangeball_contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))
                print(f"Orange Ball Cartesian Coordinates: {cartesian_coords}")
            # Tegner
            cv2.rectangle(output_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_Image, "1", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cartesian_coords, output_Image


         @staticmethod
    def convert_balls_to_cartesian(image, ball_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()

        cartesian_coords = []
        output_Image = image.copy()
        print(f"Found {len(ball_contours)} balls.")
        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords.append(ImageProcessor.convert_to_cartesian((center_x, center_y)))
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

            cv2.rectangle(output_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_Image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return cartesian_coords, output_Image


        @staticmethod
    def convert_robot_to_cartesian(output_image, robot_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()
        robot_coordinates = []

        if robot_contours is None or len(robot_contours) < 3:
            print("Not enough blue dots found.")
            return robot_coordinates, output_image  
        
        for cnt in robot_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_image, (cX, cY), 3, (0, 255, 0), -1)  # Mark the blue dots on the image
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian((cX, cY))
                    robot_coordinates.append(cartesian_coords)
                    #print(f"Robot Cartesian Coordinates: {cartesian_coords}")


        return robot_coordinates, output_image
'''


'''
scale factor 
@staticmethod
    def calculate_scale_factors():
        bottom_left, bottom_right, top_left, top_right = ImageProcessor.corners.values()
        bottom_width = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        top_width = np.linalg.norm(np.array(top_left) - np.array(top_right))
        left_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        right_height = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        x_scale = 166 / max(bottom_width, top_width)
        y_scale = 121 / max(left_height, right_height)
        return x_scale, y_scale    
'''

'''
process image block


   #GKOPERGPOKEROPKG KOM NU JOHAN VS CODE STINKER
    @staticmethod
    def process_robot(indput_Image,output_Image):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()
        midtpunkt = None
        angle = None

        contours = ImageProcessor.find_robot(indput_Image, min_size=0, max_size=100000)
       
        cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image,contours)
        
        if(contours is not None):
            midtpunkt,angle,output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
        
        
        return midtpunkt, angle, output_Image
    
        
@staticmethod
    def process_image(image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        edges = cv2.Canny(red, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = max(contours, key=cv2.contourArea)
        max_contour_area = cv2.contourArea(max_contour) * 0.99
        min_contour_area = cv2.contourArea(max_contour) * 0.002

        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]

        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 100)

        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
              ImageProcessor.corners['bottom_left'], ImageProcessor.corners['bottom_right'], \
        ImageProcessor.corners['top_left'], ImageProcessor.corners['top_right'] = \
            ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])

        # Calculate scale factors and update class attributes
        ImageProcessor.scale_factors['x_scale'], ImageProcessor.scale_factors['y_scale'] = \
            ImageProcessor.calculate_scale_factors()
        
        ImageProcessor.print_corner_info()

        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(image, [approx], 0, (60, 0, 0), 5)



        # robot_contour = ImageProcessor.find_robot(image, min_size=0, max_size=100000)
        # robot_coordinates = []

        # if robot_contour is not None:
        #     print("Found robot.")
        #     # Approximere konturen til en polygon og finde hjørnerne (spidserne)
        #     epsilon = 0.025 * cv2.arcLength(robot_contour, True)
        #     approx = cv2.approxPolyDP(robot_contour, epsilon, True)

        #     # Use k-means clustering to find the three most distinct points
        #     from sklearn.cluster import KMeans
        #     if len(approx) > 3:
        #         kmeans = KMeans(n_clusters=3)
        #         kmeans.fit(approx.reshape(-1, 2))
        #         points = kmeans.cluster_centers_.astype(int)
        #     else:
        #         points = approx

        #     for point in points:
        #         cv2.circle(image, tuple(point[0]), 5, (0, 255, 0), -1)
        #         if bottom_left_corner is not None:
        #             cartesian_coords = ImageProcessor.convert_to_cartesian(tuple(point[0]), bottom_left_corner,
        #                                                                    bottom_right_corner, top_left_corner,
        #                                                                    top_right_corner)
        #             robot_coordinates.append(cartesian_coords)
        #             print(f"Robot Kartesiske Koordinater: {cartesian_coords}")
        #     x, y, w, h = cv2.boundingRect(robot_contour)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        #     cv2.putText(image, "Robot", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # else:
        #     print("Ingen robot fundet.")

        #draw balls
        
        
        ball_contours = ImageProcessor.find_balls_hsv1(image, min_size=1000, max_size=2000)

        ImageProcessor.paintballs(ball_contours, "ball",image)

        roboball = ImageProcessor.find_robot(image, min_size=100, max_size=100000)
        ImageProcessor.paintballs(roboball, "robo ball", image)
 
        midpoint, direction = ImageProcessor.find_direction(roboball)
        if roboball and len(roboball) == 3:
            if midpoint and direction:
                    # Draw the direction from the midpoint
                endpoint = (midpoint[0] + direction[0], midpoint[1] + direction[1])
                cv2.circle(image, midpoint, 10, (0, 0, 255), -1)  # Red dot at midpoint
                cv2.line(image, midpoint, endpoint, (255, 0, 0), 3)  # Blue line indicating direction
                midpoint_cm=ImageProcessor.convert_to_cartesian(midpoint)
                print("Midpoint:", midpoint_cm)
                print("Direction to third point:", direction)
                # cv2.imshow('Directional Image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        else:
            print("Could not find exactly three balls., found ", len(roboball))       

        orange_ballcontours = ImageProcessor.find_orangeball_hsv(image, min_size=1000, max_size=2000)
        if orange_ballcontours:
            contour = orange_ballcontours[0]
            print(contour)
            ImageProcessor.paintballs(contour,"orangeball", image)


        big_ball_contour = ImageProcessor.find_bigball_hsv(image, min_size=4000, max_size=10000)
        if big_ball_contour is not None:
            ImageProcessor.paintballs(orange_ballcontours, "egg", image) 

        angle = ImageProcessor.calculate_angle(direction)
        
        if angle is not None:
            print(f"Angle: {angle} degrees")
        cross_contours = ImageProcessor.find_cross_contours(filtered_contours,image)
        for i, cnt in enumerate(cross_contours):
            cv2.drawContours(image, [cnt], 0, (255, 0, 0), 3)
            for point in cnt:
                x, y = point.ravel()
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1])) for point in cnt]

                #print(f"Cross {i+1} Cartesian Coordinates: {cartesian_coords}")

        print(f"Found {len(cross_contours)} crosses.")

        # Mark the corners on the output_image for the arena
        if bottom_left_corner is not None:
            cv2.circle(image, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(image, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(image, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(image, top_right_corner, 10, (255, 0, 135), -1)

        # cv2.imshow('image2', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        @staticmethod
        def process_robot(indput_Image, output_Image):
            midtpunkt = None
            angle = None
            contours = ImageProcessor.find_robot(indput_Image)
            cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image, contours,)
            if (contours is not None):
                midtpunkt, angle, output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
            return midtpunkt, angle, output_Image

    @staticmethod
    def process_robotForTesting(indput_Image, output_Image):
        midtpunkt = None
        angle = None
        contours = ImageProcessor.find_robot(indput_Image)
        cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image, contours)
        ImageProcessor.paintballs(contours, "robot", image)
        if (contours is not None):
            midtpunkt, angle, output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
        return midtpunkt, angle, output_Image, contours
    
    
   
    ### TEST FUNKTION FOR AT SE OM DET VIRKER
    def process_image(image):
        success,outputimage, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, filtered_contours = ImageProcessor.find_Arena(
            image, image.copy())
        if not success:
            print("Could not find the arena.")
            return
        ImageProcessor.showimage('arena', outputimage)
        
        cross_counters, output_image_with_cross = ImageProcessor.find_cross_contours( filtered_contours, outputimage)
        cartesian_cross_list, output_image_with_cross = ImageProcessor.convert_cross_to_cartesian(cross_counters, outputimage)
        # Create the mask using the detected arena corners
        ImageProcessor.showimage('cross', outputimage)

        arenaCorners = [bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner]

        balls_contour = ImageProcessor.find_balls_hsv1(outputimage, 1000,2000)
        ball_list, outputimage = ImageProcessor.convert_balls_to_cartesian(outputimage,balls_contour)
        outputimage = ImageProcessor.paintballs(balls_contour, "ball", outputimage)
        ImageProcessor.showimage('balls', outputimage)
   

        #midtpunkt, angle, output_image_with_robot, contours = ImageProcessor.process_robotForTesting(output_image_with_balls,
        #                                                                                            output_image_with_balls.copy())
        ImageProcessor.process_robotForTesting(outputimage, outputimage)
        ImageProcessor.showimage('robot', outputimage)
        
        orangeball_contour = ImageProcessor.find_orangeball_hsv(outputimage, 1000,2000)
        if(orangeball_contour):
            outputimage=ImageProcessor.paintballs(orangeball_contour, "orange", outputimage)
            ImageProcessor.showimage('orange', outputimage)


        egg_contour = ImageProcessor.find_bigball_hsv(outputimage, 5000,10000,0.3,1.8)
        if(egg_contour):
            outputimage=ImageProcessor.paintballs(egg_contour, "egg", outputimage)
            ImageProcessor.showimage('final', outputimage)

            

            @staticmethod
    def createMask(imageToDetectOn, points):
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(imageToDetectOn.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return mask

    @staticmethod
    def useMask(imageToMask, mask):
        if mask.shape[:2] != imageToMask.shape[:2]:
            raise ValueError("The mask and the image must have the same dimensions")
        return cv2.bitwise_and(imageToMask, imageToMask, mask=mask)

'''













'''

import cv2
import numpy as np
#import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load from {image_path}")
            return None
        return image

    @staticmethod
    def find_balls_hsv(image, min_size=500, white_area_size=2000, padding=15, min_size2=400):
        def detect_balls_original_mask(hsv_image, white_lower, white_upper):
            # Threshhold the HSV image to get only white colors
            white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

            # Use morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Display the white mask
            cv2.imshow('Processed white mask', white_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            ball_contours = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                print("contour area:", {area})

                if min_size < area < 10000:
                    if area > white_area_size:
                        print("entering multiple balls")
                        # Extract the region of interest
                        x, y, w, h = cv2.boundingRect(cnt)
                        x_pad = max(x - padding, 0)
                        y_pad = max(y - padding, 0)
                        w_pad = min(w + 2 * padding, image.shape[1] - x_pad)
                        h_pad = min(h + 2 * padding, image.shape[0] - y_pad)
                        sub_image = white_mask[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

                        # sure background area
                        sure_bg = cv2.dilate(sub_image, kernel, iterations=3)

                        # Distance transform
                        dist = cv2.distanceTransform(sub_image, cv2.DIST_L2, 0)

                        # foreground area
                        ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
                        sure_fg = sure_fg.astype(np.uint8)

                        # unknown area
                        unknown = cv2.subtract(sure_bg, sure_fg)

                        # Marker labelling
                        ret, markers = cv2.connectedComponents(sure_fg)
                        markers += 1
                        sub_image_color = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2BGR)
                        markers = cv2.watershed(sub_image_color, markers)
                        markers[markers == -1] = 0
                        sub_image_color[markers > 1] = [0, 165, 255]

                        hsv_image = cv2.cvtColor(sub_image_color, cv2.COLOR_BGR2HSV)
                        orange_lower = np.array([15, 100, 20], dtype="uint8")
                        orange_upper = np.array([25, 255, 255], dtype="uint8")
                        orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

                        sub_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for sub_cnt in sub_contours:
                            sub_area = cv2.contourArea(sub_cnt)
                            if min_size2 >= sub_area:
                                perimeter = cv2.arcLength(sub_cnt, True)
                                if perimeter == 0:
                                    continue
                                circularity = 4 * np.pi * (sub_area / (perimeter * perimeter))
                                if 0.7 <= circularity <= 1.2 and sub_area > 100:
                                    sub_cnt = sub_cnt + np.array([[x_pad, y_pad]])
                                    ball_contours.append(sub_cnt)
                    else:
                        print("entering single ball")
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter == 0:
                            continue
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if 0.7 <= circularity <= 1.2:
                            ball_contours.append(cnt)
            return ball_contours, white_mask

        def remove_duplicate_contours(contours):
            centroids = []
            unique_contours = []

            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))

            for i, cnt1 in enumerate(contours):
                unique = True
                for j, cnt2 in enumerate(unique_contours):
                    if i != j:
                        M1 = cv2.moments(cnt1)
                        M2 = cv2.moments(cnt2)
                        if M1["m00"] != 0 and M2["m00"] != 0:
                            cX1 = int(M1["m10"] / M1["m00"])
                            cY1 = int(M1["m01"] / M1["m00"])
                            cX2 = int(M2["m10"] / M2["m00"])
                            cY2 = int(M2["m01"] / M2["m00"])
                            if np.sqrt((cX1 - cX2) ** 2 + (cY1 - cY2) ** 2) < 10:  # distance threshold
                                unique = False
                                break
                if unique:
                    unique_contours.append(cnt1)
            return unique_contours

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initial white mask range
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        ball_contours, white_mask = detect_balls_original_mask(hsv_image, white_lower, white_upper)

        # Check if the number of detected balls is less than 12
        if len(ball_contours) < 12:
            # Adjust white mask range
            white_lower = np.array([0, 0, 245], dtype="uint8")
            white_upper = np.array([180, 60, 255], dtype="uint8")
            additional_ball_contours, additional_white_mask = detect_balls_original_mask(hsv_image, white_lower, white_upper)
            ball_contours.extend(additional_ball_contours)

        # Remove duplicate contours
        ball_contours = remove_duplicate_contours(ball_contours)

        # Display the white mask
        cv2.imshow('Processed Image Balls', white_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Draw contours on the original image
        output_image = image.copy()
        cv2.drawContours(output_image, ball_contours, -1, (0, 255, 0), 2)

        return ball_contours, output_image

    @staticmethod
    def find_orangeball_hsv(image, min_size=300, max_size=1000000000):
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for orange color in HSV
        orange_lower = np.array([15, 100, 20], dtype="uint8")
        orange_upper = np.array([30, 255, 255], dtype="uint8")

        # Threshhold the HSV image to get only white colors
        orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet
        cv2.imshow('Processed Image Orange ball', orange_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orangeball_contours = []
        # Logikken for at finde countours på boldene
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    orangeball_contours.append(cnt)
        output_image = image.copy()
        cv2.drawContours(output_image, orangeball_contours, -1, (0, 255, 0), 2)

        return orangeball_contours, output_image

    @staticmethod
    def find_robot(image, min_size=0, max_size=100000):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        output_image = image.copy()

        blue_lower = np.array([100, 85, 85], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")
        #blue_lower = np.array([105, 100, 100], dtype="uint8")
        #blue_upper = np.array([131, 255, 255], dtype="uint8")
        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('Processed Image Robot', blue_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, output_image

        robot_counters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 <= circularity <= 1.2:  # Filter round shapes based on circularity
                robot_counters.append(cnt)

        if len(robot_counters) == 0:
            print("No round contours found.")
            return None, output_image

        # Sort the round contours by area and select the three largest
        robot_counters = sorted(robot_counters, key=cv2.contourArea, reverse=True)[:3]

        return robot_counters, output_image

    @staticmethod
    def convert_to_cartesian(pixel_coords, bottom_left, bottom_right, top_left, top_right):
        x_scale = 180 / max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])
        y_scale = 120 / max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])
        x_cartesian = (pixel_coords[0] - bottom_left[0]) * x_scale
        y_cartesian = 120 - (pixel_coords[1] - top_left[1]) * y_scale
        x_cartesian = max(min(x_cartesian, 180), 0)
        y_cartesian = max(min(y_cartesian, 120), 0)
        return x_cartesian, y_cartesian

    @staticmethod
    def detect_all_corners(filtered_contours, image_width, image_height):
        corners = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            corners.extend(approx)

        corners.sort(key=lambda point: point[0][0] + point[0][1])
        top_left_corner = corners[0][0]
        bottom_left_corner = corners[1][0]
        top_right_corner = corners[2][0]
        bottom_right_corner = corners[3][0]
        top_left_corner = (max(0, top_left_corner[0]), max(0, top_left_corner[1]))
        bottom_left_corner = (max(0, bottom_left_corner[0]), min(image_height, bottom_left_corner[1]))
        top_right_corner = (min(image_width, top_right_corner[0]), max(0, top_right_corner[1]))
        bottom_right_corner = (min(image_width, bottom_right_corner[0]), min(image_height, bottom_right_corner[1]))
        return bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner

    @staticmethod
    def calculate_scale_factors(bottom_left, bottom_right, top_left, top_right):
        bottom_width = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        top_width = np.linalg.norm(np.array(top_left) - np.array(top_right))
        left_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        right_height = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        x_scale = 180 / max(bottom_width, top_width)
        y_scale = 120 / max(left_height, right_height)
        return x_scale, y_scale

    @staticmethod
    def find_cross_contours(filtered_contours, image):
        found_cross = False
        cross_contours = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True) #justere efter billede
            print(f"Contour length: {len(approx)}")  # Debug statement
            if len(approx) == 12:  # Our cross has 12 corners.
                bounding_rect = cv2.boundingRect(cnt)
                aspect_ratio = bounding_rect[2] / float(bounding_rect[3])
                print(f"Aspect ratio: {aspect_ratio}")  # Debug statement
                if 0.8 <= aspect_ratio <= 1.2:  # Bounds
                    if not found_cross:  # Locate it.
                        cross_contours.append(approx)
                        found_cross = True  # The cross got found!
                        for i, point in enumerate(approx):
                            x, y = point.ravel()
                            cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        break  # Stop searching after a cross once found.
        if not found_cross:
            print("Cross not found.")  # Debug statement
        output_image = image.copy()
        return cross_contours, output_image

    @staticmethod
    def find_arena(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(red, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Show the mask used to find the arena contours
        cv2.imshow('Arena Mask', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(contours) == 0:
            print("No contours found in arena.")  # Debug statement
            return [], image

        max_contour = max(contours, key=cv2.contourArea)
        max_contour_area = cv2.contourArea(max_contour) * 0.99
        min_contour_area = cv2.contourArea(max_contour) * 0.002

        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]
        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(image, [approx], 0, (60, 0, 0), 5)
        output_image = image.copy()
        cv2.drawContours(output_image, filtered_contours, -1, (60, 0, 0), 3)

        print(f"Filtered contours: {len(filtered_contours)}")  # Debug statement
        return filtered_contours, output_image


    @staticmethod
    def process_robot(indput_Image, output_Image, bottom_left_corner, bottom_right_corner, top_right_corner,
                      top_left_corner):
        midtpunkt = None
        angle = None
        contours = ImageProcessor.find_robot(indput_Image, output_Image)

        cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image, contours,
                                                                                   bottom_left_corner,
                                                                                   bottom_right_corner,
                                                                                   top_right_corner, top_left_corner)

        if (contours is not None):
            midtpunkt, angle, output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)


        return midtpunkt, angle, output_Image


    @staticmethod
    def draw_Goals(image, cm_start, cm_end, bottom_left, bottom_right, top_left, top_right):
        # Convert cm coordinates to pixel coordinates, so it can draw between the Y-axis between corners,
        start_pixel = ImageProcessor.convert_to_pixel(cm_start, bottom_left, bottom_right, top_left, top_right)
        end_pixel = ImageProcessor.convert_to_pixel(cm_end, bottom_left, bottom_right, top_left, top_right)

        # Draw at pixel coord.
        cv2.rectangle(image, start_pixel, end_pixel, (0, 255, 0), 4)



        return image

    @staticmethod
    def draw_midpointGoal(image, cm_center, bottom_left, bottom_right, top_left, top_right):

        cm_center = ImageProcessor.convert_to_pixel(cm_center, bottom_left, bottom_right, top_left, top_right)


        cv2.circle(image, cm_center,10, (255, 0, 0), -1)

        return image

    @staticmethod
    def routeRobotGoalSmall(image, DropPositionGoalsmall, bottom_left, bottom_right, top_left, top_right):
        DropPositionGoalsmall = ImageProcessor.convert_to_pixel(DropPositionGoalsmall, bottom_left, bottom_right, top_left, top_right)
        cv2.circle(image, DropPositionGoalsmall, 10, (255, 205, 0), -1)

        text = "GoalSmallDrop"
        text_position = (DropPositionGoalsmall[0] + 15, DropPositionGoalsmall[1])
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        ##Implement Angle Desired, and - from current robot's angle.
        #Small is 180


        return image



    @staticmethod
    def routeRobotGoalBig(image, DropPositionGoalbig, bottom_left, bottom_right, top_left, top_right):

        DropPositionGoalbig = ImageProcessor.convert_to_pixel(DropPositionGoalbig, bottom_left, bottom_right, top_left, top_right)
        cv2.circle(image, DropPositionGoalbig, 10, (255, 205, 0), -1)
        text = "GoalBigDrop"
        text_position = (DropPositionGoalbig[0] + 15, DropPositionGoalbig[1])
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        return image




    @staticmethod
    def convert_to_pixel(cm_coords, bottom_left, bottom_right, top_left, top_right):
        # Only usuable for Goal definition (reverting CM's to pixel alignment for UI)!
        x_scale = max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0]) / 166.7
        y_scale = max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1]) / 121

        x_pixel = int(cm_coords[0] * x_scale + bottom_left[0])
        y_pixel = int((121 - cm_coords[1]) * y_scale + top_left[1])

        return x_pixel, y_pixel

    def compare_angles(angleRobot, angleGoalDesired):
        angleRobot = angleRobot % 360
        angleGoalDesired = angleGoalDesired % 360
        diff = abs(angleRobot - angleGoalDesired)
        if diff > 180:
            diff = 360 - diff
        return diff

    @staticmethod
    def convert_cross_to_cartesian(cross_contours, image, bottom_left_corner, bottom_right_corner, top_left_corner,
                                   top_right_corner):
        cartesian_coords_list = []
        output_image = image.copy()

        for i, cnt in enumerate(cross_contours):
            for point in cnt:
                x, y = point.ravel()
                cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1]), bottom_left_corner,
                                                                        bottom_right_corner, top_left_corner,
                                                                        top_right_corner) for point in cnt]
                cartesian_coords_list.append(cartesian_coords)
                print(f"Cross {i + 1} Cartesian Coordinates:")
                for coord in cartesian_coords:
                    print(f"    {coord}")
        return cartesian_coords_list, output_image
    @staticmethod
    def convert_balls_to_cartesian(image,ball_contours,bottom_left_corner,bottom_right_corner,top_right_corner,top_left_corner):
        cartesian_coords_list =[]
        output_image = image.copy()

        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return cartesian_coords_list, output_image

    @staticmethod
    def convert_robot_to_cartesian(image, robot_contours, bottom_left_corner, bottom_right_corner, top_left_corner,
                                   top_right_corner):
        robot_coordinates = []
        output_image = image.copy()

        if robot_contours is None or len(robot_contours) < 3:
            print("Not enough blue dots found.")
            return robot_coordinates, output_image

        for cnt in robot_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_image, (cX, cY), 5, (0, 255, 0), -1)  # Mark the blue dots on the image
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian((cX, cY), bottom_left_corner,
                                                                           bottom_right_corner, top_left_corner,
                                                                           top_right_corner)
                    robot_coordinates.append(cartesian_coords)
                    print(f"Robot Cartesian Coordinates: {cartesian_coords}")

        return robot_coordinates, output_image

    @staticmethod
    def convert_orangeball_to_cartesian(image, orangeball_contours, bottom_left_corner, bottom_right_corner, top_right_corner,
                                        top_left_corner):
        output_image = image.copy()
        cartesian_coords = None

        if orangeball_contours:
            contour = orangeball_contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                print(f"Orange Ball Cartesian Coordinates: {cartesian_coords}")

            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, "1", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cartesian_coords, output_image

    @staticmethod
    def process_image(image):
        filtered_contours, output_image = ImageProcessor.find_arena(image)

        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
            ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])

        if bottom_left_corner is None or bottom_right_corner is None or top_left_corner is None or top_right_corner is None:
            print("Could not detect all corners.")
            return

        x_scale, y_scale = ImageProcessor.calculate_scale_factors(bottom_left_corner, bottom_right_corner,
                                                                  top_left_corner, top_right_corner)
        print("Bottom Left Corner - Pixel Coordinates:", bottom_left_corner)
        print("Bottom Left Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        print("Bottom Right Corner - Pixel Coordinates:", bottom_right_corner)
        print("Bottom Right Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        print("Top Left Corner - Pixel Coordinates:", top_left_corner)
        print("Top Left Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        print("Top Right Corner - Pixel Coordinates:", top_right_corner)
        print("Top Right Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        cross_contours, image_with_cross = ImageProcessor.find_cross_contours(filtered_contours, output_image)
        cartesian_coords_list, image_with_cross = ImageProcessor.convert_cross_to_cartesian(cross_contours,
                                                                                            output_image,
                                                                                            bottom_left_corner,
                                                                                            bottom_right_corner,
                                                                                            top_left_corner,
                                                                                            top_right_corner)

        # Mark the corners on the output_image
        if bottom_left_corner is not None:
            cv2.circle(output_image, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(output_image, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(output_image, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(output_image, top_right_corner, 10, (255, 0, 135), -1)

        ball_contours, image_with_balls = ImageProcessor.find_balls_hsv(output_image)
        cartesian_coords, image_with_balls = ImageProcessor.convert_balls_to_cartesian(output_image, ball_contours,
                                                                                       bottom_left_corner,
                                                                                       bottom_right_corner,
                                                                                       top_right_corner,
                                                                                       top_left_corner)

        robot_contours, image_with_robot = ImageProcessor.find_robot(output_image, min_size=0, max_size=100000)



        if robot_contours is not None:
            robot_coordinates, image_with_robot = ImageProcessor.convert_robot_to_cartesian(output_image,
                                                                                            robot_contours,
                                                                                            bottom_left_corner,
                                                                                            bottom_right_corner,
                                                                                            top_left_corner,
                                                                                            top_right_corner)

        #Goal positions, 1 is small, 2 is big.
        cm_position_1_start = (0, 57)
        cm_position_1_midpoint = (0, 61.5)
        cm_position_1_end = (0, 66)
        cm_position_2_start = (166, 53)
        cm_position_2_midpoint = (166, 60.9)
        cm_position_2_end = (166, 68.8)

        cm_position_goaldrop_small = (12, 61.5)
        cm_position_goaldrop_big = (154, 61)

        output_image = ImageProcessor.draw_Goals(output_image, cm_position_1_start, cm_position_1_end,
                                                 bottom_left_corner, bottom_right_corner, top_left_corner,
                                                 top_right_corner)
        output_image = ImageProcessor.draw_Goals(output_image, cm_position_2_start, cm_position_2_end,
                                                 bottom_left_corner, bottom_right_corner, top_left_corner,
                                                 top_right_corner)

        output_image = ImageProcessor.draw_midpointGoal(output_image, cm_position_1_midpoint,
                                                        bottom_left_corner, bottom_right_corner, top_left_corner,
                                                        top_right_corner)

        output_image = ImageProcessor.draw_midpointGoal(output_image, cm_position_2_midpoint,
                                                        bottom_left_corner, bottom_right_corner, top_left_corner,
                                                        top_right_corner)
        output_image = ImageProcessor.routeRobotGoalSmall(output_image, cm_position_goaldrop_small,
                                                          bottom_left_corner, bottom_right_corner, top_left_corner,
                                                          top_right_corner)

        output_image = ImageProcessor.routeRobotGoalBig(output_image, cm_position_goaldrop_big,
                                                        bottom_left_corner, bottom_right_corner, top_left_corner,
                                                        top_right_corner)


        orangeball_contours, image_with_orangeballs = ImageProcessor.find_orangeball_hsv(output_image, min_size=300,
                                                                                          max_size=1000)
        cartesian_coords_orange, image_with_orangeballs = ImageProcessor.convert_orangeball_to_cartesian(output_image,
                                                                                                         orangeball_contours,
                                                                                                         bottom_left_corner,
                                                                                                         bottom_right_corner,
                                                                                                         top_right_corner,
                                                                                                         top_left_corner)

        cv2.imshow('Final Image with cross', image_with_cross)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Final Image with Balls and Arena', image_with_balls)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Final Image with Robot', image_with_robot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Final Image with Orange Balls', image_with_orangeballs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "main/billede6(bold i hjørnet).png"  # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
'''