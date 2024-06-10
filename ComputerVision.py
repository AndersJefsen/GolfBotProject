import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image

def find_balls_hsv(image, min_size=300, max_size=1000):  # Størrelsen af hvid, der skal findes
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white color in HSV
    white_lower = np.array([0, 0, 200], dtype="uint8")
    white_upper = np.array([180, 60, 255], dtype="uint8")

    # Threshold the HSV image to get only white colors
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    # Use morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Processed Image', white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours
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

def find_blue_contours(image, min_size=300, max_size=1000):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color in HSV
    blue_lower = np.array([100, 150, 0], dtype="uint8")
    blue_upper = np.array([140, 255, 255], dtype="uint8")

    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Use morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_size <= area <= max_size:
            blue_contours.append(cnt)

    return blue_contours


def image_to_cartesian(image_point, origin):  # Funktionen som converter til koordinater
    x, y = image_point
    origin_x, origin_y = origin
    cartesian_x = x - origin_x
    cartesian_y = origin_y - y  # Invert the y-axis
    return cartesian_x, cartesian_y

def process_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Edge detection using Canny
    edges = cv2.Canny(red, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    # For the map
    max_contour_area = cv2.contourArea(max_contour) * 0.99  # remove largest except all other 99% smaller
    min_contour_area = cv2.contourArea(max_contour) * 0.002  # smaller contours

    filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]

    # Draw filtered contours on original image
    result = image.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # Initialize variables to store the extreme points
    min_x = float('inf')
    max_y = -1
    bottom_left_corner = None

    for cnt in filtered_contours:

        font = cv2.FONT_HERSHEY_COMPLEX

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv2.drawContours(image, [approx], 0, (255, 0, 0), 5)

        # Used to flat the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel()
        i = 0

        for j in n:
            if i % 2 == 0:
                x = n[i]
                y = n[i + 1]

                # Update the min_x and max_y for the bottom left corner detection
                if y > max_y or (y == max_y and x < min_x):
                    min_x = x
                    max_y = y
                    bottom_left_corner = (x, y)

            i = i + 1

    # find balls and process each contour
    ball_contours = find_balls_hsv(image, min_size=300, max_size=1000)
    print(f"Found {len(ball_contours)} balls initially.")

    for i, contour in enumerate(ball_contours, 1):
        # Compute the bounding rectangle for each ball contour
        x, y, w, h = cv2.boundingRect(contour)
        # Compute the center of the ball
        center_x = x + w // 2
        center_y = y + h // 2
        # Draw the rectangle and number on the ball
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Get the cartesian coordinates
        if bottom_left_corner is not None:  # Ensure the bottom left corner was found
            cartesian_coords = image_to_cartesian((center_x, center_y), bottom_left_corner)
            # Now you can use `cartesian_coords` as needed
            print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

    # Draw a circle at the detected bottom left corner
    if bottom_left_corner is not None:
        cv2.circle(image, bottom_left_corner, 10, (0, 0, 255), -1)

    # Showing the final image.
    cv2.imshow('image2', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/Users/peterhannibalhildorf/PycharmProjects/GolfBotProject/images/Bane 4 3 ugers/WIN_20240605_10_26_58_Pro.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
