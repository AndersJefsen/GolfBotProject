import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image

def draw_dot(image, position, label, color=(0, 255, 0), thickness=3):
    cv2.circle(image, position, 5, color, -1)
    cv2.putText(image, label, (position[0] + 10, position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def process_image(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    th = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Edge detection using Canny
    edges = cv2.Canny(th, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    min_contour_area = cv2.contourArea(max_contour) * 0.99  # remove largest except all other 99% smaller

    # Filter contours based on area (remove small noisy contours)
    # min_contour_area = 1300000 Adjust as needed
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_contour_area]

    # Draw filtered contours on original image
    result = image.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)


    cv2.imshow('Processed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # create copy of original image
    img1 = image.copy()
    # highlight white region with different color
    #img1[th == 255] = (255, 255, 0)
    #img1[th1 == 255] = (0, 255, 255)

    # Show the processed image
    cv2.imshow('Processed Image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 3 med Gule/WIN_20240207_09_25_55_Pro.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
