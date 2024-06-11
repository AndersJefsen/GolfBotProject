import cv2

import numpy as np



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
    def find_balls_hsv(input_image, min_size=50, max_size=200):
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # Define range for white color in HSV
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")

        # Threshhold the HSV image to get only white colors
        print("Thresholding image")
        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
        print("after thres")
        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet
        '''
        cv2.imshow('Processed Image', white_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # Find contours
        print("Finding contours")
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        return ball_contours

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

    #PETERS VERSION
    @staticmethod
    def find_robot(indput_Image, output_image, min_size=0, max_size=100000):
       
        hsv_image = cv2.cvtColor(indput_Image, cv2.COLOR_BGR2HSV)
      
        blue_lower = np.array([105, 100, 100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        """ koden for at se masken bliver brugt
        cv2.imshow('Processed Image Robot', blue_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

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
            return None

        # Sort the round contours by area and select the three largest
        robot_counters = sorted(robot_counters, key=cv2.contourArea, reverse=True)[:3]
     

        return robot_counters


    """
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
    @staticmethod
    def find_direction(contours):
        if len(contours) != 3:
           # print("Error: Expected exactly three contours.")
            return None, None
   
        # Use the first point of each contour directly
        points = [contour[0][0] for contour in contours]  # Take the first point from each contour
        #points = contours
        
        # Calculate pairwise distances and find the shortest
        dists = []
        pairs = []
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                dists.append(dist)
                pairs.append((points[i], points[j]))
        
        # Find the shortest edge
        min_dist_index = np.argmin(dists)
        point_a, point_b = pairs[min_dist_index]

        # Calculate the midpoint of this shortest edge
        midpoint = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)

        # Identify the third point
        third_point = next(p for p in points if not (np.array_equal(p, point_a) or np.array_equal(p, point_b)))

        # Compute direction vector from the midpoint to the third point
        direction_vector = (third_point[0] - midpoint[0], third_point[1] - midpoint[1])

        return midpoint, direction_vector

    @staticmethod
    def calculate_angle(direction_vector):
    # Calculate angle in radians
        angle_radians = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Convert angle to degrees
        angle_degrees = np.degrees(angle_radians)
        
        # Normalize angle to be between 0 and 360
        angle_degrees = angle_degrees % 360
        
        return angle_degrees

    @staticmethod
    def calculate_robot_midpoint_and_angle(cartesian_coords, output_Image):
        if len(cartesian_coords) != 3:
            print("Error: Expected exactly three coordinates for the robot.")
            return None, None, output_Image
        
        midpoint, direction = ImageProcessor.find_direction(cartesian_coords)
      
        if midpoint and direction:
            # Draw the direction from the midpoint
            endpoint = (int(midpoint[0] + direction[0]), int(midpoint[1] + direction[1]))
            cv2.circle(output_Image, (int(midpoint[0]), int(midpoint[1])), 10, (0, 0, 255), -1)  # Red dot at midpoint
            cv2.line(output_Image, (int(midpoint[0]), int(midpoint[1])), endpoint, (255, 0, 0),
                     3)  # Blue line indicating direction

            angle = ImageProcessor.calculate_angle(direction)
            return midpoint, angle, output_Image

        return None, None, output_Image

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
            approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)  # justere efter billede
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
    def find_Arena(inputImage,outPutImage):
        lab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(red, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
       
    
       
        if not contours:
            print("No contours found")
            return False, None, None, None, None, None

        max_contour = max(contours, key=cv2.contourArea)
        #max_contour_area = cv2.contourArea(max_contour) * 0.99
        #min_contour_area = cv2.contourArea(max_contour) * 0.002
        max_contour_area = 1000000
        min_contour_area = 5000


        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if max_contour_area > area > min_contour_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:  # Only consider contours with exactly 4 points
                    filtered_contours.append(approx)

        print(f"Total contours found: {len(contours)}")
        print(f"Filtered contours: {len(filtered_contours)}")
        for cnt in filtered_contours:
            print(f"Contour area: {cv2.contourArea(cnt)}")
        
        if len(filtered_contours) < 1:
            print("Not enough filtered contours")
            return False, None, None, None, None, None
        
        cv2.drawContours(outPutImage, filtered_contours, -1, (0, 255, 0), 2)
    
        try:
            bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
                ImageProcessor.detect_all_corners(filtered_contours, inputImage.shape[1], inputImage.shape[0])
        except ValueError as e:
            print(f"Error in detecting corners: {e}")
            return False, None, None, None, None, None
        x_scale, y_scale = ImageProcessor.calculate_scale_factors(bottom_left_corner, bottom_right_corner,
                                                                  top_left_corner, top_right_corner)
        
        print("Bottom Left Corner - Pixel Coordinates:", bottom_left_corner)
        print("Bottom Left Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Bottom Right Corner - Pixel Coordinates:", bottom_right_corner)
        print("Bottom Right Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Top Left Corner - Pixel Coordinates:", top_left_corner)
        print("Top Left Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Top Right Corner - Pixel Coordinates:", top_right_corner)
        print("Top Right Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))
        
        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(inputImage, [approx], 0, (60, 0, 0), 5)

        if bottom_left_corner is not None:
            cv2.circle(outPutImage, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(outPutImage, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(outPutImage, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(outPutImage, top_right_corner, 10, (255, 0, 135), -1)

        if (bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner) is not None:
            return True,outPutImage,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner
        else:
            return False, None, None, None, None, None

    @staticmethod
    def convert_balls_to_cartesian(image, ball_contours, bottom_left_corner, bottom_right_corner, top_right_corner,
                                   top_left_corner):
        cartesian_coords_list = []
        output_Image = image.copy()

        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

            cv2.rectangle(output_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_Image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return cartesian_coords_list, output_Image

    @staticmethod
    def convert_robot_to_cartesian(output_image, robot_contours, bottom_left_corner, bottom_right_corner, top_left_corner,
                                   top_right_corner):
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
                    cartesian_coords = ImageProcessor.convert_to_cartesian((cX, cY), bottom_left_corner,
                                                                           bottom_right_corner, top_left_corner,
                                                                           top_right_corner)
                    robot_coordinates.append(cartesian_coords)
                    print(f"Robot Cartesian Coordinates: {cartesian_coords}")


        return robot_coordinates, output_image

    @staticmethod
    def convert_cross_to_cartesian(cross_contours, image, bottom_left_corner, bottom_right_corner, top_left_corner,
                                   top_right_corner):
        cartesian_coords_list = []
        output_Image = image.copy()

        for i, cnt in enumerate(cross_contours):
            for point in cnt:
                x, y = point.ravel()
                #Tegner
                cv2.circle(output_Image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1]), bottom_left_corner,
                                                                        bottom_right_corner, top_left_corner,
                                                                        top_right_corner) for point in cnt]
                cartesian_coords_list.append(cartesian_coords)
                print(f"Cross {i + 1} Cartesian Coordinates:")
                for coord in cartesian_coords:
                    print(f"    {coord}")
        return cartesian_coords_list, output_Image

    @staticmethod
    def convert_orangeball_to_cartesian(image, orangeball_contours, bottom_left_corner, bottom_right_corner,
                                        top_right_corner,
                                        top_left_corner):
        output_Image = image.copy()
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
            # Tegner
            cv2.rectangle(output_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_Image, "1", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cartesian_coords, output_Image

    #GKOPERGPOKEROPKG KOM NU JOHAN VS CODE STINKER
    @staticmethod
    def process_robot(indput_Image,output_Image,bottom_left_corner, bottom_right_corner,top_right_corner,top_left_corner):
        midtpunkt = None
        angle = None
        contours = ImageProcessor.find_robot(indput_Image, output_Image)
        print("contours works")
        cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image,contours,bottom_left_corner, bottom_right_corner,top_right_corner,top_left_corner)
        print("cartesian works")
        print(cartesian_coords)
        if(contours is not None):
            midtpunkt,angle,output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
        print("midtpunkt works")

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
            ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])
        x_scale, y_scale = ImageProcessor.calculate_scale_factors(bottom_left_corner, bottom_right_corner,
                                                                  top_left_corner, top_right_corner)
        print("Bottom Left Corner - Pixel Coordinates:", bottom_left_corner)
        print("Bottom Left Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Bottom Right Corner - Pixel Coordinates:", bottom_right_corner)
        print("Bottom Right Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Top Left Corner - Pixel Coordinates:", top_left_corner)
        print("Top Left Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Top Right Corner - Pixel Coordinates:", top_right_corner)
        print("Top Right Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

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


        roboball = ImageProcessor.find_robot(image, min_size=50, max_size=100000)
        for i, contour in enumerate(roboball, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                print(f"roboBall {i} Cartesian Coordinates: {cartesian_coords}")
 
        midpoint, direction = ImageProcessor.find_direction(roboball)
        if roboball and len(roboball) == 3:
            if midpoint and direction:
                    # Draw the direction from the midpoint
                endpoint = (midpoint[0] + direction[0], midpoint[1] + direction[1])
                cv2.circle(image, midpoint, 10, (0, 0, 255), -1)  # Red dot at midpoint
                cv2.line(image, midpoint, endpoint, (255, 0, 0), 3)  # Blue line indicating direction
                print("Midpoint:", midpoint)
                print("Direction to third point:", direction)
                cv2.imshow('Directional Image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Could not find exactly three balls., found ", len(roboball))        

        angle = ImageProcessor.calculate_angle(direction)
        
        if angle is not None:
            print(f"Angle: {angle} degrees")
        cross_contours = ImageProcessor.find_cross_contours(filtered_contours)
        for i, cnt in enumerate(cross_contours):
            cv2.drawContours(image, [cnt], 0, (255, 0, 0), 3)
            for point in cnt:
                x, y = point.ravel()
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1]), bottom_left_corner,
                                                                        bottom_right_corner, top_left_corner,
                                                                        top_right_corner) for point in cnt]

                print(f"Cross {i+1} Cartesian Coordinates: {cartesian_coords}")

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

        ball_contours = ImageProcessor.find_balls_hsv(image, min_size=300, max_size=1000)
        orange_ballcontours = ImageProcessor.find_orangeballs_hsv(image, min_size=300, max_size=1000)

        ball_contours = ImageProcessor.find_balls_threshold(image, min_size=300, max_size=1000)
        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

        # Process orange ball contour
        if orange_ballcontours:
            contour = orange_ballcontours[0]
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Orange Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                               bottom_right_corner, top_left_corner,
                                                                               top_right_corner)
                print(f"Orange Ball Cartesian Coordinates: {cartesian_coords}")


        big_ball_contour = ImageProcessor.find_bigball_threshold(image, min_size=1000, max_size=5000)
        if big_ball_contour is not None:
            x, y, w, h = cv2.boundingRect(big_ball_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "EGG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                               bottom_right_corner, top_left_corner,
                                                               top_right_corner)
        print(f"Big Ball Cartesian Coordinates: {cartesian_coords}")




        cv2.imshow('image2', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    image_path = "images/Bane 4 3 ugers/3ball.jpg"  # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
