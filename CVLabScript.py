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

    # Apply thresholding
    th1 = cv2.threshold(lab[:, :, 0], 200, 255, cv2.THRESH_BINARY)[1]

    edges = cv2.Canny(th, 200, 400)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # find the largest contour, ie the map
        epsilon = 0.01 * cv2.arcLength(largest_contour,True)
        approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)  # makes largest contour into main corners

        if len(approx_corners) == 4:
            cv2.polylines(image, [approx_corners], True, (255, 0, 0), 3)

            # calculate center of map
            center_x = int(sum([corner[0][0] for corner in approx_corners]) / 4)
            center_y = int(sum([corner[0][1] for corner in approx_corners]) / 4)
            center = (center_x, center_y)

            # find and label corners
            sorted_corners = sorted(approx_corners[:, 0, :], key=lambda x: (x[1], x[0]))
            top_corners = sorted(sorted_corners[:2], key=lambda x: x[0])
            bottom_corners = sorted(sorted_corners[2:], key=lambda x: x[0], reverse=True)

            # corner labels
            draw_dot(image, tuple(top_corners[0]), 'TL (0,120)')
            draw_dot(image, tuple(top_corners[1]), 'TR (180,120)')
            draw_dot(image, tuple(bottom_corners[0]), 'BR (180,0)')
            draw_dot(image, tuple(bottom_corners[1]), 'BL (0,0)')
            draw_dot(image, center, 'Center')

    # create copy of original image
    img1 = image.copy()
    # highlight white region with different color
    img1[th == 255] = (255, 255, 0)
    img1[th1 == 255] = (0, 255, 255)

    # Show the processed image
    cv2.imshow('Processed Image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 3 med Gule/WIN_20240207_09_22_48_Pro.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)