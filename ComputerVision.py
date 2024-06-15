import cv2
import numpy as np
import math
class ImageProcessor:
    corners = {'bottom_left': None, 'bottom_right': None, 'top_left': None, 'top_right': None}
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
            cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))

            # Display the Cartesian coordinates on the image
            text_location = (x, y + h + 20)  # Display below the contour
            cv2.putText(output_image, f"Coords: {cartesian_coords}", text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
    def find_balls_hsv1(image, min_size=300, white_area_size=1000, padding=15, min_size2=400, max_size=10000):
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
    def find_robot(indput_Image, min_size=50, max_size=100000):
       
      
        blue_lower = np.array([105, 100, 100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")
        blue_mask=ImageProcessor.apply_hsv_filter(indput_Image, blue_lower,blue_upper)

    
        blue_mask = ImageProcessor.clean_mask(blue_mask)
        """ koden for at se masken bliver brugt
        cv2.imshow('Processed Image Robot', blue_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        robot_counters=ImageProcessor.filter_circles(contours, min_size,max_size)
        if robot_counters is None:
            return None
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
            else:
                return Image

    @staticmethod
    def getrobot(coords, output_Image):

        midpoint, direction = ImageProcessor.find_direction(coords)

        
        if midpoint and direction:
            midpoint = ImageProcessor.adjust_coordinates(midpoint[0], midpoint[1], output_Image.shape[1], output_Image.shape[0])

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
    def find_cross_contours(filtered_contours, image):
        found_cross = False
        cross_contours = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)  # justere efter billede
            #print(f"Contour length: {len(approx)}")  # Debug statement
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
    '''
    @staticmethod
    def get_corrected_coordinates_robot(robot_x, robot_y, robot_z=31, cam_x=81, cam_y=61, cam_z=170):
        cam_x,cam_y=ImageProcessor.convert_to_pixel((cam_x, cam_y))
        
    # Calculate the vector from the camera to the robot on the plane
        vector_x = robot_x - cam_x
        vector_y = robot_y - cam_y
        #print(vector_x)
        # Apply the scale factor for perspective based on height
        scale_factor = robot_z / cam_z
        x_2d = cam_x + vector_x * (1 - scale_factor)
        y_2d = cam_y + vector_y * (1 - scale_factor)

        return (x_2d, y_2d)
   
    
    '''
    process image block
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
