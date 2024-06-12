import cv2
import numpy as np

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
    def filter_circles(contours, min_size, max_size, min_circularity=0.7, max_circularity=1.2):
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if min_circularity <= circularity <= max_circularity:
                filtered_contours.append(cnt)
        return filtered_contours

    @staticmethod
    def detect_and_filter_objects(image, lower_color, upper_color, min_size, max_size):
        mask = ImageProcessor.apply_hsv_filter(image, lower_color, upper_color)
        clean_mask = ImageProcessor.clean_mask(mask)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return ImageProcessor.filter_circles(contours, min_size, max_size)

    @staticmethod
    def find_orangeball_hsv(image, min_size=300, max_size=1000000000):
        orange_lower = np.array([15, 100, 20], dtype="uint8")
        orange_upper = np.array([30, 255, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, orange_lower, orange_upper, min_size, max_size)

    @staticmethod
    def find_balls_hsv(image, min_size=50, max_size=200):
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, white_lower, white_upper, min_size, max_size)
    
    @staticmethod
    def find_bigball_hsv(image, min_size=300, max_size=1000):
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, white_lower, white_upper, min_size, max_size)

    @staticmethod
    def find_robot(image, min_size=0, max_size=100000):
        blue_lower = np.array([105, 100, 100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")
        return ImageProcessor.detect_and_filter_objects(image, blue_lower, blue_upper, min_size, max_size)


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
        x_scale, y_scale = ImageProcessor.calculate_scale_factors()
        
        ImageProcessor.print_corner_info()

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
    #convert to cartesian    
    @staticmethod
    def convert_to_cartesian(pixel_coords):
        bottom_left, bottom_right, top_left, top_right = ImageProcessor.corners.values()
        x_scale, y_scale = ImageProcessor.scale_factors['x_scale'], ImageProcessor.scale_factors['y_scale']
        x_scale = 180 / max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])
        y_scale = 120 / max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])
        x_cartesian = (pixel_coords[0] - bottom_left[0]) * x_scale
        y_cartesian = 120 - (pixel_coords[1] - top_left[1]) * y_scale
        x_cartesian = max(min(x_cartesian, 180), 0)
        y_cartesian = max(min(y_cartesian, 120), 0)
        return x_cartesian, y_cartesian
    
    @staticmethod
    def calculate_scale_factors():
        bottom_left, bottom_right, top_left, top_right = ImageProcessor.corners.values()
        bottom_width = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        top_width = np.linalg.norm(np.array(top_left) - np.array(top_right))
        left_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        right_height = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        x_scale = 180 / max(bottom_width, top_width)
        y_scale = 120 / max(left_height, right_height)
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
            print(f"{label} {i} Cartesian Coordinates: {cartesian_coords}")

        return cartesian_coords_list, output_image
    @staticmethod
    def convert_balls_to_cartesian(image, ball_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()

        cartesian_coords = []
        output_Image = image.copy()

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

        for cnt in robot_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_image, (cX, cY), 3, (0, 255, 0), -1)  # Mark the blue dots on the image
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian((cX, cY))
                    robot_coordinates.append(cartesian_coords)
                    print(f"Robot Cartesian Coordinates: {cartesian_coords}")


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
                print(f"Cross {i + 1} Cartesian Coordinates:")
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

    #GKOPERGPOKEROPKG KOM NU JOHAN VS CODE STINKER
    @staticmethod
    def process_robot(indput_Image,output_Image):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()
        midtpunkt = None
        angle = None
        contours = ImageProcessor.find_robot(indput_Image, output_Image)
       
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
        def paintballs(countors, text):
            for i, contour in enumerate(countors, 1):
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))
        
        ball_contours = ImageProcessor.find_balls_hsv(image, min_size=300, max_size=1000)
        paintballs(ball_contours, "ball")

        roboball = ImageProcessor.find_robot(image, min_size=50, max_size=100000)
        paintballs(roboball, "robo ball")
 
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

            orange_ballcontours = ImageProcessor.find_orangeball_hsv(image, min_size=300, max_size=1000)
        if orange_ballcontours:
            contour = orange_ballcontours[0]
            paintballs(contour,"orangeball")


        big_ball_contour = ImageProcessor.find_bigball_hsv(image, min_size=1000, max_size=5000)
        if big_ball_contour is not None:
            paintballs(big_ball_contour, "egg") 

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
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1])) for point in cnt]

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

        cv2.imshow('image2', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 4 3 ugers/3ball.jpg"   # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
