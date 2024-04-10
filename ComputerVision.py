import cv2
import numpy as np


# cv2.imshow('Processed Image', th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image


# For at finde balls nemmere
def find_balls_hsv(image, min_size=20, max_size=1000):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and white
    # yellow_lower = np.array([20, 100, 100], dtype="uint8")
    # yellow_upper = np.array([30, 255, 255], dtype="uint8")

    white_lower = np.array([0, 0, 190], dtype="uint8")
    white_upper = np.array([180, 55, 255], dtype="uint8")

    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    cv2.imshow('Processed Image', white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    ball_contours = [cnt for cnt in contours if min_size <= cv2.contourArea(cnt) <= max_size]

    return ball_contours


def image_to_cartesian(image_point, origin):
    """
    Convert an image point to Cartesian coordinates with the given origin.

    Parameters:
        image_point (tuple): The point (x, y) in image coordinates.
        origin (tuple): The origin (x, y) in image coordinates.

    Returns:
        (tuple): The point in Cartesian coordinates.
    """
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

    # cv2.imshow('Processed Image', white)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    max_contour = max(contours, key=cv2.contourArea)

    # For the map
    max_contour_area = cv2.contourArea(max_contour) * 0.99  # remove largest except all other 99% smaller
    min_contour_area = cv2.contourArea(max_contour) * 0.002  # smaller contours

    filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]

    # Draw filtered contours on original image
    # result = image.copy()
    # cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # 3. forsøg på at sætte koordinater på bunden af venstre hjørne
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

                # String containing the co-ordinates.
                string = str(x) + " " + str(y)
                # Update the min_x and max_y for the bottom left corner detection
                if y > max_y or (y == max_y and x < min_x):
                    min_x = x
                    max_y = y
                    bottom_left_corner = (x, y)

            i = i + 1

    # find balls and process each contour
    ball_contours = find_balls_hsv(image, min_size=20, max_size=1000)
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

    # cv2.imshow('Processed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 3 med Gule/WIN_20240207_09_22_41_Pro.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
