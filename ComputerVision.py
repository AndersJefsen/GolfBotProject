import cv2
import numpy as np
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main')))
from data import Data 
class ImageProcessor:
    corners = {'bottom_left': None, 'bottom_right': None, 'top_left': None, 'top_right': None}
    def __init__(self):
        pass
    @staticmethod
    def find_contour_center(contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # Calculate the X coordinate of the centroid
            cY = int(M["m01"] / M["m00"])  # Calculate the Y coordinate of the centroid
            return (cX, cY)
        return None
    
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
    def draw_route_on_image(image, points, line_color=(255, 0, 0), line_thickness=2,):

        """
        Draw a route as a series of lines connecting a list of points on an existing image.

        Args:
        image (numpy.ndarray): The image on which to draw the route.
        points (list of tuple): List of points (x, y) where each point is a tuple.
        line_color (tuple): The color of the line (B, G, R).
        line_thickness (int): Thickness of the lines.
        show_image (bool): If True, display the image after drawing the route.

        Returns:
        numpy.ndarray: The image with the route drawn on it.
        """
        # Check if there are at least two points to connect
     
        if len(points) < 2:
            print("need two points")
            return image
        points = [(int(round(x)), int(round(y))) for x, y in points]

        # Draw lines between each consecutive pair of points
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], line_color, line_thickness)
            

        # Optionally display the image
      

        return image
    @staticmethod
    def show_contours_with_areas(image, contours, window_name="Contours with Areas"):
        black_background = np.zeros_like(image)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            cv2.drawContours(black_background, [cnt], -1, (0, 255, 0), 3)  # green contour line

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.putText(black_background, f"Area: {area}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        ImageProcessor.showimage(window_name, black_background)

    @staticmethod
    def showimage(name="pic", image=None):
        try:
            cv2.imshow(name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"error {e} with pic {name}")   

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
    def paint_and_print_contours(image, contours):
        output_image = image.copy()  # Work on a copy of the image to preserve the original

        for i, cnt in enumerate(contours, 1):
            # Calculate the contour area to avoid processing empty contours
            if cv2.contourArea(cnt) == 0:
                continue

            # Draw each contour on the output image
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)  # Green for visibility

            # Find the bounding rectangle to get the center
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_y = y + h // 2

            # Convert to Cartesian coordinates if needed
            #cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))

            # Display the Cartesian coordinates on the image
            text_location = (x, y + h + 20)  # Display below the contour
            cv2.putText(output_image, f"Coords: {(center_x, center_y)}", text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_image

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
    def find_bigball_hsv(image, min_size=2000, max_size=8000, min_curvature=0.7, max_curvature=1.2):
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, white_lower, white_upper, min_size, max_size, min_curvature, max_curvature)

    @staticmethod
    def find_balls_hsv1(image, min_size=200, white_area_size=1000, padding=15, min_size2=400, max_size=10000):
        def detect_balls_original_mask(hsv_image, white_lower, white_upper):
            # Threshhold the HSV image to get only white colors
            white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

            # Use morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            ball_contours = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
               # print("contour area:", {area})

                if min_size < area < max_size:
                    if area > white_area_size:
                       # print("entering multiple balls")
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

                        ImageProcessor.filter_circles(sub_contours,min_size,max_size)            
                    else:
                        #print("entering single ball")
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
                centroids.append(ImageProcessor.find_contour_center(cnt))

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
        if ball_contours is not None:
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
    def find_robot(indput_Image, min_size=100, max_size=2000):
       

        # blue_lower = np.array([80, 66, 100], dtype="uint8")
        #blue_lower = np.array([85, 100, 100], dtype="uint8")
        #blue_upper = np.array([131, 255, 255], dtype="uint8")
        #blue_mask=ImageProcessor.apply_hsv_filter(indput_Image, blue_lower,blue_upper)
        # Når robotten har grønne cirkler

        green_lower = np.array([36, 25, 25], dtype="uint8")
        green_upper = np.array([86, 255, 255], dtype="uint8")

        green_mask = ImageProcessor.apply_hsv_filter(indput_Image, green_lower, green_upper)

        green_mask = ImageProcessor.clean_mask(green_mask)

        #blue_mask = ImageProcessor.clean_mask(blue_mask)
        #koden for at se masken bliver brugt
        """
        cv2.imshow('Processed Image Robot', green_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        robot_counters=ImageProcessor.filter_circles(contours, min_size,max_size)
        if robot_counters is None:
            print("robot not found")
            return None
        if len(robot_counters) <3:
            print("Robot not found, Not enough contours found.")
            return None

        # Sort the round contours by area and select the three largest
        robot_counters = sorted(robot_counters, key=cv2.contourArea, reverse=True)[:3]

        return robot_counters

    @staticmethod
    def remove_cross(contours, image, expansion_factor):
        if contours is None:
            print("No contours provided.")
            return image

        # Create a mask of the same size as the image, initialized to all white (255)
        mask = np.ones(image.shape[:2], dtype="uint8") * 255

        for contour in contours:
            # Ensure the contour is a numpy array
            if isinstance(contour, np.ndarray):
                # Find the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Expand the bounding box to cover the desired area
                x_start = max(int(x - w * (expansion_factor - 1) / 2), 0)
                y_start = max(int(y - h * (expansion_factor - 1) / 2), 0)
                x_end = min(int(x + w * (1 + (expansion_factor - 1) / 2)), image.shape[1])
                y_end = min(int(y + h * (1 + (expansion_factor - 1) / 2)), image.shape[0])

                # Draw a filled rectangle over the expanded bounding box
                cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (0), thickness=cv2.FILLED)
            else:
                print(f"Unexpected contour type: {type(contour)}")

        # Apply the mask to the image: where mask is 0 (the cross area), set the image to black
        result_image = cv2.bitwise_and(image, image, mask=mask)

        return result_image
    @staticmethod
    def mask_out_arena_corners(corners,image,mask_size):
        # Create a mask of the same size as the image, initialized to all white (255)
        mask = np.ones(image.shape[:2], dtype="uint8") * 255

        # Define the corners
        bottom_left, bottom_right, top_left, top_right = corners

        # Define the triangles to mask each corner
        triangles = [
            [bottom_left, (bottom_left[0] + mask_size, bottom_left[1]), (bottom_left[0], bottom_left[1] - mask_size)],
            # Bottom left
            [bottom_right, (bottom_right[0] - mask_size, bottom_right[1]),
             (bottom_right[0], bottom_right[1] - mask_size)],  # Bottom right
            [top_left, (top_left[0] + mask_size, top_left[1]), (top_left[0], top_left[1] + mask_size)],  # Top left
            [top_right, (top_right[0] - mask_size, top_right[1]), (top_right[0], top_right[1] + mask_size)]  # Top right
        ]

        # Correcting the triangles' directions
        triangles = [
            [bottom_left, (bottom_left[0] + mask_size, bottom_left[1]), (bottom_left[0], bottom_left[1] - mask_size)],
            # Bottom left
            [bottom_right, (bottom_right[0] - mask_size, bottom_right[1]),
             (bottom_right[0], bottom_right[1] - mask_size)],  # Bottom right
            [top_left, (top_left[0] - mask_size, top_left[1]), (top_left[0], top_left[1] + mask_size)],  # Top left
            [top_right, (top_right[0] + mask_size, top_right[1]), (top_right[0], top_right[1] + mask_size)]  # Top right
        ]

        # Draw the triangles on the mask
        for triangle in triangles:
            pts = np.array(triangle, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (0))

        # Apply the mask to the image: where mask is 0 (the corner areas), set the image to black
        result_image = cv2.bitwise_and(image, image, mask=mask)

        return result_image

   

    @staticmethod
    def find_direction(contours):
        if len(contours) != 3:
           # print("Error: Expected exactly three contours.")
            return None, None
   
        # Use the first point of each contour directly
        points=[]
        for cnt in contours:
            points.append(ImageProcessor.find_contour_center(cnt))  # Take the first point from each contour
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
            else:
                return Image

    @staticmethod
    def getrobot(coords, output_Image):

        midpoint, direction = ImageProcessor.find_direction(coords)

        
        if midpoint and direction:

            angle = ImageProcessor.calculate_angle(direction)
            return midpoint, angle, output_Image, direction

        return None, None, output_Image, None

    @staticmethod
    def paintrobot(midpoint,angle,output_image,direction):
        cv2.putText(output_image, f"Angle: {angle:.2f}", (int(midpoint[0]) + 20, int(midpoint[1]) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        cv2.circle(output_image, (int(midpoint[0]), int(midpoint[1])), 10, (0, 0, 255), -1)  # Red dot at midpoint
        endpoint = (int(midpoint[0] + direction[0]), int(midpoint[1] + direction[1]))

        cv2.line(output_image, (int(midpoint[0]), int(midpoint[1])), endpoint, (255, 0, 0),3)
        return output_image


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
    def find_cross_contours(input_image):
        # Convert the image to LAB color space
        lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

        # Apply threshold to the 'A' channel to find the red areas
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Perform edge detection
        edges = cv2.Canny(red, 100, 200)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Display the mask used for edge detection
        """
        cv2.imshow('Cross', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        cross_contours = []
       # print(f"Total contours found: {len(contours)}")  # Debug statement
        found_cross = False
        for cnt in contours:
            bounding_rect = cv2.boundingRect(cnt)
            aspect_ratio = bounding_rect[2] / float(bounding_rect[3])
            area = cv2.contourArea(cnt)
            #print(f"Contour aspect ratio: {aspect_ratio}")  # Debug statement
            if 0.8 <= aspect_ratio <= 1.2 and area > 1000:  # Aspect ratio bounds for a cross-like shape
                cross_contours.append(cnt)
                found_cross = True
                #print("Cross contour found")
                break  # Stop searching after a cross is found

        if not found_cross:
            print("Cross not found.")  # Debug statement'
            return None

        return cross_contours
    @staticmethod
    def find_cross_corners(contour):
        for cnt in contour:
            epsilon = 0.014 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            iterations = 0
            max_iterations = 20

            while len(approx) != 12 and iterations < max_iterations:
                # Update epsilon value to find 12 corners
                if len(approx) > 12:
                    epsilon += 0.0005 * cv2.arcLength(cnt, True)
                else:
                    epsilon -= 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                iterations += 1

            if len(approx) == 12:
                #print(f"Cross corners found {len(approx)} after {iterations} iterations.")
                return approx
           # else:
               # print(f"Found {len(approx)} corners after {iterations} iterations, could not find exactly 12 corners.")

        return None  # Return None if no contour with exactly 12 corners is found

    @staticmethod
    def draw_cross_corners(image,corners):
    # Draw circles at each corner with the specified color
        for point in corners:
            x, y = point.ravel()
            cv2.circle(image, (x, y), 5, (128, 0, 128), -1)  # Purple color circles
        return image


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
        x_scale = 166 / max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])
        y_scale = 121 / max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])
        x_cartesian = (pixel_coords[0] - bottom_left[0]) * x_scale
        y_cartesian = 121 - (pixel_coords[1] - top_left[1]) * y_scale
        x_cartesian = max(min(x_cartesian, 166), 0)
        y_cartesian = max(min(y_cartesian, 121), 0)
        return x_cartesian, y_cartesian
    
    def show_graph(image, valid_paths):
    # Draw all valid paths
        for path, _ in valid_paths:
            print(path)
            for i in range(len(path) - 1):
                cv2.line(image, (int(path[i][0]), int(path[i][1])),
                        (int(path[i+1][0]), int(path[i+1][1])), (255, 255, 0), 2)  # Yellow lines for paths

        # Highlight the nodes (optional, if needed)
        nodes = set()
        for path, _ in valid_paths:
            nodes.update(path)
        for node in nodes:
            cv2.circle(image, (int(node[0]), int(node[1])), 5, (0, 0, 255), -1)  # Red nodes

        return image
    
    @staticmethod
    def process_and_convert_contours(image, contours, label="pic"):
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
            #cv2.putText(output_image, f"{label} {cartesian_coords}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #print(f"{label} {i} Cartesian Coordinates: {cartesian_coords}")

        return cartesian_coords_list, output_image
        


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
    def convert_to_pixel(cm_coords):
        bottom_left, bottom_right, top_left, top_right=ImageProcessor.corners.values()
        # Only usuable for Goal definition (reverting CM's to pixel alignment for UI)!
        x_scale = max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0]) / 166.7
        y_scale = max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1]) / 121

        x_pixel = int(cm_coords[0] * x_scale + bottom_left[0])
        y_pixel = int((121 - cm_coords[1]) * y_scale + top_left[1])

        return x_pixel, y_pixel
    
    @staticmethod
    def compare_angles(angleRobot, angleGoalDesired):
        angleRobot = angleRobot % 360
        angleGoalDesired = angleGoalDesired % 360
        diff = abs(angleRobot - angleGoalDesired)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    
    

    @staticmethod
     #johan
    def get_corrected_coordinates_robot(cX, cY,data: Data, adjustment_factor=0.2):
        # Calculate the center of the area
        height, width, _ = data.screenshot.shape
        centerX = width / 2
        centerY = height / 2
        
        # Calculate vector from the point to the center
        vector_to_center_x = centerX - cX
        vector_to_center_y = centerY - cY
        
        # Calculate the distance from the point to the center
        distance_to_center = math.sqrt(vector_to_center_x**2 + vector_to_center_y**2)
        
        # Normalize the vector to the center
        if distance_to_center != 0:  # Prevent division by zero
            normalized_vector = (vector_to_center_x / distance_to_center, vector_to_center_y / distance_to_center)
        else:
            normalized_vector = (0, 0)
        
        # Scale the normalized vector by the adjustment factor
        adjustment_x = normalized_vector[0] * adjustment_factor * distance_to_center
        adjustment_y = normalized_vector[1] * adjustment_factor * distance_to_center
        
        # Adjust coordinates
        new_cX = cX + adjustment_x
        new_cY = cY + adjustment_y
        
        return new_cX, new_cY
    '''
    @staticmethod
    def get_corrected_coordinates_robot(x,y):
        
        camera_height= 170
        robot_height = 31
        cam_x=81
        cam_y=61
        camera_x,camera_y=ImageProcessor.convert_to_pixel((cam_x, cam_y))

        # Calculate distance from camera x,y to robot x,y (what the camera sees)
        distance = math.sqrt((x - camera_x) ** 2 + (y - camera_y) ** 2)
        #pythagoras to find the hypotenuse
        hypotenuse = math.sqrt(distance ** 2 + camera_height ** 2)

        #find smalle triangle
        h= camera_height-robot_height
        factor_triangle=h/camera_height

        l=distance*factor_triangle

        small_l = distance-l
        #print(x,y)
        
        # Correcting the x-coordinate
        if x > camera_x:
            corrected_x = x - (small_l)  # Move right
        else:
            corrected_x = x + (small_l)  # Move left

        # Correcting the y-coordinate
        if y > camera_y:
            corrected_y = y - (small_l)  # Move up
        else:
            corrected_y = y + (small_l)  # Move down
            
        corrected_coordinates=[corrected_x,corrected_y]

        return corrected_coordinates
   
    @staticmethod
    def get_corrected_coordinates_robot(x,y,roboth=31,camerah=170):
        
        
       

        # Calculate distance from camera x,y to robot x,y (what the camera sees)
        scale_factor = roboth / camerah
        print(scale_factor)
        x_2d = x + (scale_factor * x)
        y_2d = y + (scale_factor * y)

        return (x_2d, y_2d)
    

    @staticmethod
    #anders
    def get_corrected_coordinates_robot(x, y, data: Data, robot_height=31, camera_height=165):

        height, width, _ = data.screenshot.shape

        cam_x, cam_y = width // 2, height // 2
        cam_x,cam_y =ImageProcessor.convert_to_cartesian((cam_x,cam_y))
        x,y =ImageProcessor.convert_to_cartesian((x,y))

        B = math.sqrt((cam_x - x)**2 + (cam_y - y)**2)
        
        if B == 0:
            return x, y
        # Beregn den vandrette afstand x fra punkt P til robotten
        x_robot = (B * robot_height) / camera_height
        
        # Beregn interpolationen for x og y koordinater
        R_x = x + (x_robot / B) * (cam_x - x)
        R_y = y + (x_robot / B) * (cam_y - y)
        return ImageProcessor.convert_to_pixel((R_x, R_y))
    

    
    @staticmethod
    #victor
    def get_corrected_coordinates_robot(robot_x, robot_y, data: Data, robot_z=31, cam_z=165 ):
        height, width, _ = data.screenshot.shape

        cam_x, cam_y = width // 2, height // 2

    # Calculate the vector from the camera to the robot on the plane
        vector_x = robot_x - cam_x
        vector_y = robot_y - cam_y
        #print(vector_x)
        # Apply the scale factor for perspective based on height
        scale_factor = robot_z / cam_z
        x_2d = cam_x + vector_x * (1 - scale_factor)
        y_2d = cam_y + vector_y * (1 - scale_factor)

        return (x_2d, y_2d)
        
   
    
    
    
    @staticmethod
    def get_corrected_coordinates_robot(robot_x, robot_y, data: Data, robot_z=31, cam_z=165 ):
        height, width, _ = data.screenshot.shape

        cam_x, cam_y = width // 2, height // 2
    # Calculate the vector from the camera to the robot on the plane
        B_x = robot_x - cam_x
        B_y = robot_y - cam_y
        #print(vector_x)
        # Apply the scale factor for perspective based on height
        x_2d = robot_x+ (B_x * robot_z)/cam_x
        y_2d = robot_y+ (B_y * robot_z)/cam_y

        return (x_2d, y_2d)
    
    '''


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
