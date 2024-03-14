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

    if filtered_contours:
        # Sort contours by area to get the second largest contour
        sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        if len(sorted_contours) > 1:
            largest_contour = sorted_contours[0]
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx_corners) == 4:
                cv2.polylines(image, [approx_corners], True, (255, 0, 0), 3)

                # Calculate the bounding box of the contour
                x, y, w, h = cv2.boundingRect(approx_corners)

                # Define the number of divisions for the grid
                rows = 10
                cols = 10

                # Calculate the step size for each division
                step_x = w // cols
                step_y = h // rows

                # Draw grid lines and label coordinates
                for i in range(rows):
                    for j in range(cols):
                        # Calculate coordinates of grid points
                        grid_point_x = x + step_x * (j + 0.5)  # Add 0.5 to get the center of the grid cell
                        grid_point_y = y + step_y * (i + 0.5)
                        grid_point = (int(grid_point_x), int(grid_point_y))
                        # Draw grid points
                        cv2.circle(image, grid_point, 3, (0, 255, 0), -1)
                        # Label coordinates
                        cv2.putText(image, f'({grid_point_x}, {grid_point_y})', (int(grid_point_x), int(grid_point_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                # Show the image
                cv2.imshow('Processed Image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    cv2.imshow('Processed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # create copy of original image
    # img1 = image.copy()
    # highlight white region with different color
    # img1[th == 255] = (255, 255, 0)
    # img1[th1 == 255] = (0, 255, 255)


if __name__ == "__main__":
    image_path = "images/Bane 2 uden gule/WIN_20240207_09_37_09_Pro.jpg"  # Path to your image
    image = cv2.imread(image_path)
    if image is not None:
        process_image(image)
