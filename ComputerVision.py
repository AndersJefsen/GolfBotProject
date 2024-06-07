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
    def scale_contour(contour, scale=1.2):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        scaled_contour = []
        for point in contour:
            x, y = point[0]
            x = cx + scale * (x - cx)
            y = cy + scale * (y - cy)
            scaled_contour.append([[int(x), int(y)]])
        return np.array(scaled_contour, dtype=np.int32)

    @staticmethod
    def find_balls_hsv(image, min_size=300, max_size=1000000000):
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for white color in HSV
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")

        # Threshhold the HSV image to get only white colors
        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet
        cv2.imshow('Processed Image', white_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
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
    def find_orangeballs_hsv(image, min_size=300, max_size=1000000000):
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for orange color in HSV
        orange_lower = np.array([150, 50, 20], dtype="uint8")
        orange_upper = np.array([250, 200, 90], dtype="uint8")

        # Threshhold the HSV image to get only white colors
        orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet
        cv2.imshow('Processed Image', orange_mask)
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
        return orangeball_contours

    @staticmethod
    def find_robot(image, min_size=0, max_size=100000):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([111, 100, 100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")

        # Threshhold the HSV image image to get only blue colors
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        # Load maskerne på billedet
        cv2.imshow('Processed Image', blue_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        scaled_contour = ImageProcessor.scale_contour(largest_contour, scale=1.2)
        print(f"Largest contour area: {cv2.contourArea(largest_contour)}")

        area = cv2.contourArea(scaled_contour)
        if min_size <= area <= max_size:
            return scaled_contour
        return None

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
    def find_cross_contours(contours):
        cross_contours = []
        found_cross = False
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 12:  # our Cross  has 12 corner.
                bounding_rect = cv2.boundingRect(cnt)
                aspect_ratio = bounding_rect[2] / float(bounding_rect[3])
                if 0.8 <= aspect_ratio <= 1.2:  #Bounds
                    if not found_cross:  # Locate it.
                        cross_contours.append(approx)
                        found_cross = True  # the cross got found folks!
                        for i, point in enumerate(approx):
                            x, y = point.ravel()
                            cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        break  #stop searching after a cross once found.
        return cross_contours


    @staticmethod
    def find_Arena(inputImage,outPutImage):
        lab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(red, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        max_contour_area = cv2.contourArea(max_contour) * 0.99
        min_contour_area = cv2.contourArea(max_contour) * 0.002
        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]
        
        cv2.drawContours(outPutImage, filtered_contours, -1, (0, 255, 0), 2)

        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
            ImageProcessor.detect_all_corners(filtered_contours, inputImage.shape[1], inputImage.shape[0])
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



        robot_contour = ImageProcessor.find_robot(image, min_size=0, max_size=100000)
        robot_coordinates = []

        if robot_contour is not None:
            print("Found robot.")
            # Approximere konturen til en polygon og finde hjørnerne (spidserne)
            epsilon = 0.025 * cv2.arcLength(robot_contour, True)
            approx = cv2.approxPolyDP(robot_contour, epsilon, True)

            # Use k-means clustering to find the three most distinct points
            from sklearn.cluster import KMeans
            if len(approx) > 3:
                kmeans = KMeans(n_clusters=3)
                kmeans.fit(approx.reshape(-1, 2))
                points = kmeans.cluster_centers_.astype(int)
            else:
                points = approx

            for point in points:
                cv2.circle(image, tuple(point[0]), 5, (0, 255, 0), -1)
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian(tuple(point[0]), bottom_left_corner,
                                                                           bottom_right_corner, top_left_corner,
                                                                           top_right_corner)
                    robot_coordinates.append(cartesian_coords)
                    print(f"Robot Kartesiske Koordinater: {cartesian_coords}")
            x, y, w, h = cv2.boundingRect(robot_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(image, "Robot", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            print("Ingen robot fundet.")

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

        if bottom_left_corner is not None:
            cv2.circle(image, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(image, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(image, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(image, top_right_corner, 10, (255, 0, 135), -1)

        ball_contours = ImageProcessor.find_balls_hsv(image, min_size=300, max_size=1000)
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

        cv2.imshow('image2', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    image_path = "images/Bane 4 3 ugers/Johansbillede_3uger.jpg"  # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
