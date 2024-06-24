import numpy as np
import cv2 as cv


def createMask(imageToDetectOn,points):
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros(imageToDetectOn.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [points], 255)

    return mask

def useMask(imageToMask,mask):
    if mask.shape[:2] != imageToMask.shape[:2]:
        raise ValueError("The mask and the image must have the same dimensions")
    
    return cv.bitwise_and(imageToMask, imageToMask, mask=mask)
def drawHelpPoints(output_Image, points, color=(0, 0, 255)):
    for point in points:
        #print("point: ", point)
        # Ensure the point is a flat list or tuple with two elements
        if isinstance(point, (list, tuple)) and len(point) == 2:
            # Convert the point coordinates to integers
            center = (int(point[0]), int(point[1]))
            cv.circle(output_Image, center, 5, color, -1)
        else:
            print(f"Unexpected point format: {point}")
    return output_Image

def drawLine(output_image, point1, point2, color=(0, 255, 0)):
    if isinstance(point1, (list, tuple, np.ndarray)) and isinstance(point2, (list, tuple, np.ndarray)):
        point1 = [int(point1[0]), int(point1[1])]
        point2 = [int(point2[0]), int(point2[1])]
        cv.line(output_image, (point1[0], point1[1]), (point2[0], point2[1]), color, 2)
    else:
        print(f"Unexpected point format: {point1} or {point2}")
    return output_image
def drawArea(output_image, points, color=(0, 255, 0)):
    if len(points) > 1:
        for i in range(len(points) - 1):
            output_image = drawLine(output_image, points[i], points[i + 1],color)
        output_image = drawLine(output_image, points[-1], points[0],color)
    return output_image