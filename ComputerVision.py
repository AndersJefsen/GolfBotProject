import cv2
import numpy as np
import socket

# Server settings
# The third number needs to be changed each time the hotspot changes
HOST = '192.168.123.243'  # The IP address of your EV3 brick
PORT = 1024  # The same port as used by the server


def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(command.encode('utf-8'))
    except ConnectionRefusedError:
        print("Could not connect to the server. Please check if the server is running and reachable.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Example usage


"""
while True:
    command = input()
    send_command(command)
"""


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

        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")

        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Processed Image', white_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                # Check for circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:  # Adjust thresholds as needed for your specific conditions
                    ball_contours.append(cnt)

        return ball_contours

    @staticmethod
    def convert_to_cartesian(pixel_coords, bottom_left, bottom_right, top_left, top_right):
        # Calculate scaling factors for x and y axes (so there can be a proportion between pixel distance).
        x_scale = 180 / max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])
        y_scale = 120 / max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])

        # Map pixel coordinates to Cartesian coordinates
        x_cartesian = (pixel_coords[0] - bottom_left[0]) * x_scale
        y_cartesian = 120 - (pixel_coords[1] - top_left[1]) * y_scale  #adjust to top-left origin

        # x-coordinate range (0 to 180)
        x_cartesian = max(min(x_cartesian, 180), 0)

        # y-coordinate range (0 to 120)
        y_cartesian = max(min(y_cartesian, 120), 0)

        return x_cartesian, y_cartesian

    @staticmethod
    def detect_all_corners(filtered_contours, image_width, image_height):
        corners = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            corners.extend(approx)

        # Sort the corners by their x and y coordinates
        corners.sort(key=lambda point: point[0][0] + point[0][1])

        # Extract the four corners(maybe this should be redone, depends on accuracy)
        top_left_corner = corners[0][0]
        bottom_left_corner = corners[1][0]
        top_right_corner = corners[2][0]
        bottom_right_corner = corners[3][0]

        # Ensure tha corners are within tha picture!
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

        # Define the Cartesian distances between corners
        x_scale = 180 / max(bottom_width, top_width)
        y_scale = 120 / max(left_height, right_height)

        return x_scale, y_scale

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
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
            ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])


        # Calculate scale factors
        x_scale, y_scale = ImageProcessor.calculate_scale_factors(bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)


        print("Bottom Left Corner - Pixel Coordinates:", bottom_left_corner)
        print("Bottom Left Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Bottom Right Corner - Pixel Coordinates:", bottom_right_corner)
        print("Bottom Right Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Top Left Corner - Pixel Coordinates:", top_left_corner)
        print("Top Left Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        print("Top Right Corner - Pixel Coordinates:", top_right_corner)
        print("Top Right Corner - Cartesian Coordinates:", (round(ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[0], 2), abs(round(ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)[1], 2))))

        # Use RGB for setting up color. (-1 er thickness)
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
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)

                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")
        # coords_str = f"{cartesian_coords[0]},{cartesian_coords[1]}"
        # send_command(coords_str)

        cv2.imshow('image2', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 4 3 ugers/WIN_20240605_10_27_29_Pro.jpg"
    #image = ImageProcessor.load_image(image_path)
    image = cv2.imread(image_path)
    if image is not None:
        ImageProcessor.process_image(image)