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


def process_image(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    white = cv2.threshold(lab[:, :, 0], 200, 255, cv2.THRESH_BINARY)[1]

    # Edge detection using Canny
    edges = cv2.Canny(red, 100, 200)



    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    balls, _ = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('Processed Image', white)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    max_contour = max(contours, key=cv2.contourArea)
    max_contour_ball = max(balls, key=cv2.contourArea)

    # For the map
    max_contour_area = cv2.contourArea(max_contour) * 0.99  # remove largest except all other 99% smaller
    min_contour_area = cv2.contourArea(max_contour) * 0.002  # smaller contours

    # For the balls
    max_ball_area = cv2.contourArea(max_contour_ball) * 0.99
    min_ball_area = cv2.contourArea(max_contour_ball) * 0.00001

    filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]

    filtered_balls = [cnt for cnt in max_contour_ball if max_ball_area > cv2.contourArea(cnt) > min_ball_area]

    # Draw filtered contours on original image
    # result = image.copy()
    # cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

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
    for cnt in filtered_balls:
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

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
    image_path = "images/banemedfarve2/banemedfarve5.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
