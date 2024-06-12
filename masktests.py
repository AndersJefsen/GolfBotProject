import cv2
import numpy as np
import os

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
    def find_balls_hsv(image, min_size=300, max_size=1000000000):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")
        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    ball_contours.append(cnt)
        output_image = image.copy()
        cv2.drawContours(output_image, ball_contours, -1, (0, 255, 0), 2)
        return ball_contours, output_image

    @staticmethod
    def find_orangeball_hsv(image, min_size=300, max_size=1000000000):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        orange_lower = np.array([15, 100, 20], dtype="uint8")
        orange_upper = np.array([30, 255, 255], dtype="uint8")
        orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orangeball_contours = []
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
        blue_lower = np.array([105, 100, 100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, image
        robot_counters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 <= circularity <= 1.2:
                robot_counters.append(cnt)
        if len(robot_counters) == 0:
            print("No round contours found.")
            return None, image
        robot_counters = sorted(robot_counters, key=cv2.contourArea, reverse=True)[:3]
        return robot_counters, image

    @staticmethod
    def find_arena(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(red, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No contours found in arena.")
            return [], image
        max_contour = max(contours, key=cv2.contourArea)
        max_contour_area = cv2.contourArea(max_contour) * 0.99
        min_contour_area = cv2.contourArea(max_contour) * 0.002
        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]
        output_image = image.copy()
        cv2.drawContours(output_image, filtered_contours, -1, (60, 0, 0), 3)
        return filtered_contours, output_image

    @staticmethod
    def process_image(image):
        filtered_contours, output_image = ImageProcessor.find_arena(image)
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])
        ball_contours, image_with_balls = ImageProcessor.find_balls_hsv(output_image, min_size=300, max_size=1000)
        orangeball_contours, image_with_orangeballs = ImageProcessor.find_orangeball_hsv(output_image, min_size=300, max_size=1000)
        robot_contours, image_with_robot = ImageProcessor.find_robot(output_image, min_size=0, max_size=100000)
        return image_with_balls, image_with_orangeballs, image_with_robot

def process_directory(mask_path, images_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, dirs, files in os.walk(images_directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                image = ImageProcessor.load_image(image_path)
                if image is not None:
                    image_with_balls, image_with_orangeballs, image_with_robot = ImageProcessor.process_image(image)
                    cv2.imwrite(os.path.join(output_directory, f"balls_{file}"), image_with_balls)
                    cv2.imwrite(os.path.join(output_directory, f"orangeballs_{file}"), image_with_orangeballs)
                    cv2.imwrite(os.path.join(output_directory, f"robot_{file}"), image_with_robot)

if __name__ == "__main__":
    mask_path = "ComputerVision_2.0.py"  # Path to your mask
    images_directory = "path/to/your/images_directory"  # Path to your images directory
    output_directory = "path/to/your/output_directory"  # Path to save the processed images
    process_directory(mask_path, images_directory, output_directory)
