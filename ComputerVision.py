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

    # Edge detection using Canny
    edges = cv2.Canny(th, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    max_contour_area = cv2.contourArea(max_contour) * 0.99 # remove largest except all other 99% smaller
    min_contour_area = cv2.contourArea(max_contour) * 0.002 # smaller contours

    filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]

    # Draw filtered contours on original image
    # result = image.copy()
    # cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # cv2.imshow('Processed Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for cnt in filtered_contours:

        font = cv2.FONT_HERSHEY_COMPLEX

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)

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
    image_path = "images/banemedfarve2/banemedfarve7.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
