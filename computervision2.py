import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image


def find_balls(image, threshold=230,min_contour_size=100):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#filter contours based on size
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_size]

    return filtered_contours


def draw_dot(image, position, label, color=(0, 255, 255)):
    cv2.circle(image, position, 5, color, -1)
    cv2.putText(image, label, (position[0] + 10, position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


#original_image_path = 'images/Bane 2 uden gule/WIN_20240207_09_37_09_Pro.jpg'
#original_image_path = 'images/Bane 2 uden gule/bane2UdenGule1.jpg'
original_image_path = 'images/Bane 2 uden gule/WIN_20240207_09_37_26_Pro.jpg'
image = load_image(original_image_path)

# edge detection to find maps edges
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
red_mask = cv2.inRange(hsv_image, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
           cv2.inRange(hsv_image, np.array([170, 120, 70]), np.array([180, 255, 255]))
edges = cv2.Canny(red_mask, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)#find the largest contour, ie the map
    epsilon = 0.009 * cv2.arcLength(largest_contour, True) #calculate epsilon for contour approximaton, because of imperfect contours
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True) #makes largest contour into main corners

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

#  minimum contour size to filter out small objects
min_contour_size = 500
# find balls and process each contour
ball_contours = find_balls(image)
for i, contour in enumerate(ball_contours, 1):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # make into  the 180x120 coordinate system where bottom left corner is 0,0
    real_x = (x - bottom_corners[1][0]) / (top_corners[1][0] - bottom_corners[1][0]) * 180
    real_y = 120 - (y - top_corners[0][1]) / (bottom_corners[1][1] - top_corners[0][1]) * 120
    print(f"Ball {i} at: ({real_x:.2f}cm, {real_y:.2f}cm) ")

cv2.imshow("", image)
cv2.waitKey(0)
cv2.destroyAllWindows()