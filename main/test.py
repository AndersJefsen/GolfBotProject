from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision

import cv2
import numpy as np
import os

def find_inner_points_at_cross(image, distance):
    # Load contours using custom functions from your ComputerVision module
    contours = ComputerVision.ImageProcessor.find_cross_contours(image)
    points = []
    for contour in contours:
        # Approximate contour to reduce number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Visualize and save contour approximation
        temp_img = image.copy()
        cv2.drawContours(temp_img, [approx], -1, (0, 255, 0), 3)
        cv2.imshow('approx_contours.jpg', temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if len(approx) >= 4:
            for i in range(len(approx)):
                # Get current corner and adjacent points
                prev = approx[i - 1][0]
                curr = approx[i][0]
                next = approx[(i + 1) % len(approx)][0]
                
                # Calculate vectors
                v1 = prev - curr
                v2 = next - curr
                
                # Calculate angle using dot product
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angle_deg = np.degrees(angle)
                
                # Check for concave angle (inner corners)
                if angle_deg > 180:
                    angle_deg = 360 - angle_deg  # Adjust angle measurement for inner corners

                if 85 <= angle_deg <= 95:
                    # Direction vector bisecting the angle (adjusted for inner corners)
                    direction = (v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2))
                    unit_direction = direction / np.linalg.norm(direction)
                    
                    # Calculate the new point
                    new_point = curr + unit_direction * (-distance)  # Move inside the cross
                    points.append(new_point)
                    
                    # Visualize the corner and the new point
                    corner_img = image.copy()
                    cv2.circle(corner_img, tuple(prev), 5, (255, 0, 0), -1)
                    cv2.circle(corner_img, tuple(curr), 5, (0, 255, 0), -1)
                    cv2.circle(corner_img, tuple(next), 5, (0, 0, 255), -1)
                    cv2.circle(corner_img, tuple(new_point.astype(int)), 5, (255, 255, 0), -1)
                    cv2.imshow(f'inner_corner_and_new_point_{i}.jpg', corner_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    return points

# Usage
image = cv2.imread('kryds 1.jpg')
distance = 10  # Distance into the corner
new_points = find_inner_points_at_cross(image, distance)

# To visualize the final points on the original image
for point in new_points:
    cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)

cv2.imwrite('final_image_with_inner_points.jpg', image)
cv2.imshow('Image with new inner points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()