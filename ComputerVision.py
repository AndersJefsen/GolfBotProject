import cv2
import numpy as np
import math
class ImageProcessor:
    corners = {'bottom_left': None, 'bottom_right': None, 'top_left': None, 'top_right': None}
    scale_factors = {'x_scale': None, 'y_scale': None}
    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load from {image_path}")
            return None
        return image
#balls
    @staticmethod
    def apply_hsv_filter(image, lower_color, upper_color):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        return mask

    @staticmethod
    def clean_mask(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def show_contours_with_areas(image, contours, window_name="Contours with Areas"):
        # Create a black image of the same dimensions as the input image
        black_background = np.zeros_like(image)

        # Draw contours and text
        for cnt in contours:
            # Calculate the contour area
            area = cv2.contourArea(cnt)

            # Draw the contour on the black background
            cv2.drawContours(black_background, [cnt], -1, (0, 255, 0), 3)  # green contour line

            # Get the bounding rect to place the text in a visible area
            x, y, w, h = cv2.boundingRect(cnt)

            # Put the area of the contour on the image
            cv2.putText(black_background, f"Area: {area}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the result in a window
        cv2.imshow(window_name, black_background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def filter_circles(contours, min_size, max_size, min_circularity=0.7, max_circularity=1.2):
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
            if min_circularity <= circularity <= max_circularity:  # Filter round shapes based on circularity
                robot_counters.append(cnt)
            
        return robot_counters 

    @staticmethod
    def showcontours(image, contours,name=None):
        black_background = np.zeros_like(image)
        cv2.drawContours(black_background, contours, -1, (0, 255, 0), 2)  # Green color for visibility
        ImageProcessor.showimage(name, black_background)

    @staticmethod
    def detect_and_filter_objects(image, lower_color, upper_color, min_size, max_size, min_curvature=0.7, max_curvature=1.2):
        mask = ImageProcessor.apply_hsv_filter(image, lower_color, upper_color)
        clean_mask = ImageProcessor.clean_mask(mask)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #ImageProcessor.show_contours_with_areas(image, contours)


        return ImageProcessor.filter_circles(contours, min_size, max_size,min_curvature, max_curvature)

    @staticmethod
    def find_orangeball_hsv(image, min_size=100, max_size=400):
        orange_lower = np.array([15, 100, 20], dtype="uint8")
        orange_upper = np.array([30, 255, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, orange_lower, orange_upper, min_size, max_size)

    @staticmethod
    def find_balls_hsv(image, min_size=100, max_size=400):
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, white_lower, white_upper, min_size, max_size)
    
    @staticmethod
    def find_bigball_hsv(image, min_size=400, max_size=2000, min_curvature=0.7, max_curvature=1.2):
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, white_lower, white_upper, min_size, max_size, min_curvature, max_curvature)

    @staticmethod
    def find_balls_hsv1(image, min_size=300, white_area_size=2000, padding=15, min_size2=400):
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
            # cv2.imshow('Processed white mask', white_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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
            additional_ball_contours, additional_white_mask = detect_balls_original_mask(hsv_image, white_lower,
                                                                                         white_upper)
            ball_contours.extend(additional_ball_contours)

        # Remove duplicate contours
        ball_contours = remove_duplicate_contours(ball_contours)

        # Display the white mask
        # cv2.imshow('Processed Image Balls', white_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Draw contours on the original image
        output_image = image.copy()
        cv2.drawContours(output_image, ball_contours, -1, (0, 255, 0), 2)

        return ball_contours
    @staticmethod
    def find_robot(indput_Image, min_size=50, max_size=100000):
       
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

        if len(robot_counters) <3:
            print("Not enough contours found.")
            return None

        # Sort the round contours by area and select the three largest
        robot_counters = sorted(robot_counters, key=cv2.contourArea, reverse=True)[:3]
     

        return robot_counters

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
    def paintballs(countors, text, Image):
            if countors:

                for i, contour in enumerate(countors, 1):
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    cv2.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(Image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    if ImageProcessor.corners['bottom_left'] is not None:
                        cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))
                return Image

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
            cv2.putText(output_Image, f"Angle: {angle:.2f}", (int(midpoint[0]) + 20, int(midpoint[1]) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
            return midpoint, angle, output_Image

        return None, None, output_Image

  

    @staticmethod
    def detect_all_corners(filtered_contours, image_width, image_height):
        corners = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
                corners.extend(approx)

        if len(corners) < 4:
            print("Not enough corners found.")
            return None, None, None, None

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
    def print_corner_info():
        for key, corner in ImageProcessor.corners.items():
            cartesian = ImageProcessor.convert_to_cartesian(corner)
            print(f"{key.title()} Corner - Pixel Coordinates:", corner)
            print(f"{key.title()} Corner - Cartesian Coordinates:",
                  (round(cartesian[0], 2), abs(round(cartesian[1], 2))))

    @staticmethod
    def find_Arena(inputImage, outPutImage):
        lab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(red, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outPutImage, contours, -1, (0, 255, 0), 2)

        # Display the mask used for edge detection
        # cv2.imshow('Arena Mask', outPutImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        """
        cv2.imshow('Arena Mask', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        if not contours:
            print("No contours found")
            return False, None, None, None, None, None, None

        max_contour = max(contours, key=cv2.contourArea)

    # Draw only the largest contour on the output image
    #     cv2.drawContours(outPutImage, [max_contour], -1, (0, 255, 0), 3)  # Draw in green

    # # Optionally display the original image and the output image with the largest contour
    #     cv2.imshow('Largest Contour', outPutImage)
    #     cv2.waitKey(0)

        max_contour_area = cv2.contourArea(max_contour) * 0.99
        #max_contour_area = min(cv2.contourArea(max_contour) * 0.95,1500000)
        min_contour_area = 50000
        #min_contour_area = max(cv2.contourArea(max_contour) * 0.1,50000)
        #max_contour_area = 1000000
        #min_contour_area = 50000

        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]
        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(inputImage, [approx], 0, (60, 0, 0), 5)
        outPutImage = inputImage.copy()
        print(f"Total contours found: {len(contours)}")
        print(f"Filtered contours: {len(filtered_contours)}")
        for cnt in filtered_contours:
            print(f"Contour area: {cv2.contourArea(cnt)}")

        if len(filtered_contours) < 1:
            print("Not enough filtered contours")
            return False, None, None, None, None, None, None

        

        # Show the image with the first contour
        # cv2.imshow("First Filtered Contour", outPutImage)
        # cv2.waitKey(0)  # Wait for a key press to proceed
        # cv2.destroyAllWindows()  # Close the image display window

        try:
            bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners[
            'bottom_left'], ImageProcessor.corners['bottom_right'], \
            ImageProcessor.corners['top_left'], ImageProcessor.corners['top_right'] = \
            ImageProcessor.detect_all_corners(filtered_contours, inputImage.shape[1], inputImage.shape[0])

        except ValueError as e:
            print(f"Error in detecting corners: {e}")
            return False, None, None, None, None, None

        ImageProcessor.print_corner_info()
        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(outPutImage, [approx], 0, (60, 0, 0), 5)

        if bottom_left_corner is not None:
            cv2.circle(outPutImage, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(outPutImage, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(outPutImage, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(outPutImage, top_right_corner, 10, (255, 0, 135), -1)

        if (bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner) is not None:
            return True, outPutImage, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, contours
        else:
            return False, None, None, None, None, None
    #convert to cartesian    
    @staticmethod
    def convert_to_cartesian(pixel_coords):
        bottom_left, bottom_right, top_left, top_right = ImageProcessor.corners.values()
        x_scale, y_scale = ImageProcessor.scale_factors['x_scale'], ImageProcessor.scale_factors['y_scale']
        x_scale = 166 / max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])
        y_scale = 121 / max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])
        x_cartesian = (pixel_coords[0] - bottom_left[0]) * x_scale
        y_cartesian = 121 - (pixel_coords[1] - top_left[1]) * y_scale
        x_cartesian = max(min(x_cartesian, 166), 0)
        y_cartesian = max(min(y_cartesian, 121), 0)
        return x_cartesian, y_cartesian
    
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
    
    @staticmethod
    def process_and_convert_contours(image, contours, label):
        output_image = image.copy()
        cartesian_coords_list = []
        
        for i, cnt in enumerate(contours, 1):
            if cv2.contourArea(cnt) == 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_y = y + h // 2

            cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))
            cartesian_coords_list.append(cartesian_coords)

            # Marking the image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #print(f"{label} {i} Cartesian Coordinates: {cartesian_coords}")

        return cartesian_coords_list, output_image
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

    @staticmethod
    def convert_cross_to_cartesian(cross_contours, image):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()
        cartesian_coords_list = []
        output_Image = image.copy()

        for i, cnt in enumerate(cross_contours):
            for point in cnt:
                x, y = point.ravel()
                #Tegner
                cv2.circle(output_Image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1])) for point in cnt]
                cartesian_coords_list.append(cartesian_coords)
                #print(f"Cross {i + 1} Cartesian Coordinates:")
                for coord in cartesian_coords:
                    print(f"    {coord}")
        return cartesian_coords_list, output_Image

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
    def get_corrected_position(before_robot_coordinates:tuple):
        x=before_robot_coordinates[0]
        y=before_robot_coordinates[1]
        map_width = 166.5
        map_height = 121.8
        robot_body_height = 30
        camera_x = 77  # centered camera position (x-coordinate)
        camera_y = 51  # centered camera position (y-coordinate)
        # distance from camera to robot base (account for camera offset)
        distance_to_base = math.sqrt((x - camera_x)**2 + (y - camera_y)**2)

        # make sure x and y are within map boundaries 
        if x < 0 or x > map_width:
            raise ValueError("X coordinate is outside map boundaries")
        if y < 0 or y > map_height:
            raise ValueError("Y coordinate is outside map boundaries")

        # calc angle between camera and top of robot
        angle = math.atan2(robot_body_height, distance_to_base)

        # Correct the x and y coordinates
        corrected_x = x + distance_to_base * math.sin(angle)
        corrected_y = y + distance_to_base * math.cos(angle)

        corrected_position_robot = [corrected_x, corrected_y]

        return corrected_position_robot


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


    @staticmethod
    def showimage(name="pic", image=None):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        outputimage=ImageProcessor.paintballs(balls_contour, "ball", outputimage)
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


if __name__ == "__main__":
    image_path = "main/peter.png"  # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)


if __name__ == "__main__":
    image_path = "peter.png"   # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
