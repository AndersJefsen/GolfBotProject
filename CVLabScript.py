import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image


def process_image(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    th = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Extract the L channel (lightness)
    l_channel = lab[:, :, 0]

    # Apply thresholding
    th1 = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)[1]


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
    image_path = "images/banemedfarve2/banemedfarve7.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)