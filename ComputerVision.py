import math

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


def find_balls(image, min_contour_size=40, max_contour_size = 1200):
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    # white = cv2.threshold(lab[:, :, 0], 190, 255, cv2.THRESH_BINARY)[1]


    # cv2.imshow('Processed Image', white)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 190], dtype="uint8")
    white_upper = np.array([180, 55, 255], dtype="uint8")
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    # cv2.imshow('Processed Image', white_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    ball_contours = [cnt for cnt in contours if min_contour_size <= cv2.contourArea(cnt) <= max_contour_size]

    return ball_contours


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

    for cnt in filtered_contours:

        font = cv2.FONT_HERSHEY_COMPLEX

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv2.drawContours(image, [approx], 0, (255, 0, 0), 5)

        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel()
        i = 0

        for j in n:
            if i % 2 == 0:
                x = n[i]
                y = n[i + 1]

                # String containing the co-ordinates.
                string = str(x) + " " + str(y)

                if i == 0:
                    # text on topmost co-ordinate.
                    cv2.putText(image, "Top left", (x, y), font, 0.5, (255, 0, 0))
                else:
                    # text on remaining co-ordinates.
                    cv2.putText(image, string, (x, y), font, 0.5, (0, 255, 0))
            i = i + 1
    # find balls and process each contour
    ball_contours = find_balls(image)
    for i, contour in enumerate(ball_contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Showing the final image.
    cv2.imshow('image2', image)

    # cv2.imshow('Processed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # create copy of original image
    # img1 = image.copy()
    # highlight white region with different color
    # img1[th == 255] = (255, 255, 0)
    # img1[th1 == 255] = (0, 255, 255)


if __name__ == "__main__":
    image_path = "images/Bane 3 med Gule/WIN_20240207_09_22_48_Pro.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
