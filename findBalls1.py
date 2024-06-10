import cv2

import numpy as np
import sklearn as sklearn

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
    def find_orangeballs_hsv(image, min_size=300, max_size=1000000000):
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for orange color in HSV
        orange_lower = np.array([10, 100, 20], dtype="uint8")
        orange_upper = np.array([25, 255, 255], dtype="uint8")

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

if __name__ == "__main__":
    image_path = "/Users/mikkel/Desktop/4.Semester/CDIO/Code/Python/images/Bane 4 3 ugers/WIN_20240605_10_31_52_Pro.jpg"  # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.find_orangeballs_hsv(image)