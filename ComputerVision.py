import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    hist = cv2.equalizeHist(gray)

    # Show the grayscale image
    #cv2.imshow('Grayscale Image', hist)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Define lower and upper bounds for red color detection
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([100, 100, 255])
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    # Thresholds for red and white colours
    red_mask = cv2.inRange(image, lower_red, upper_red)
    white_mask = cv2.inRange(image, lower_white, upper_white)

    combined_mask = cv2.bitwise_or(red_mask, white_mask)

    # Find contours in the red mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Show the grayscale image
    cv2.imshow('Grayscale Image', combined_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Draw contours on the original image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Ignore small contours
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw green contours around detected red areas

    # Show the processed image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/banemedfarve2/banemedfarve7.jpg"  # Path to your image
    image = load_image(image_path)
    if image is not None:
        process_image(image)
