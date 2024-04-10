import cv2
import numpy as np
from numpy.ma.testutils import approx


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
    def find_balls_hsv(image, min_size=20, max_size=1000):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 190], dtype="uint8")
        white_upper = np.array([180, 55, 255], dtype="uint8")

        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_contours = [cnt for cnt in contours if min_size <= cv2.contourArea(cnt) <= max_size]

        return ball_contours

    @staticmethod
    def image_to_cartesian(image_point, origin):
        x, y = image_point
        origin_x, origin_y = origin
        cartesian_x = x - origin_x
        cartesian_y = origin_y - y  # Invert the y-axis
        return cartesian_x, cartesian_y

    @staticmethod
    def detect_all_corners(filtered_contours):
        min_x = float('inf')
        max_x = -1
        min_y = float('inf')
        max_y = -1
        bottom_left_corner = None
        bottom_right_corner = None
        top_left_corner = None
        top_right_corner = None

        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            n = approx.ravel()
            i = 0

            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i + 1]

                    if x < min_x:
                        min_x = x
                        bottom_left_corner = (x, y)
                    if x > max_x:
                        max_x = x
                        bottom_right_corner = (x, y)
                    if y < min_y:
                        min_y = y
                        top_left_corner = (x, y)
                    if y > max_y:
                        max_y = y
                        top_right_corner = (x, y)

                i += 1

        return bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner

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


        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
        ImageProcessor.detect_all_corners(filtered_contours)


#Use RGB for setting up color.
        if bottom_left_corner is not None:
            cv2.circle(image, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(image, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(image, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(image, top_right_corner, 10, (255, 0, 135), -1)

        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(image, [approx], 0, (255, 0, 0), 5)

        # after detecting them balls , process each contour
        ball_contours = ImageProcessor.find_balls_hsv(image, min_size=20, max_size=1000)
        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.image_to_cartesian((center_x, center_y), bottom_left_corner)
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

        cv2.imshow('image2', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 3 med Gule/WIN_20240207_09_22_48_Pro.jpg"
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
