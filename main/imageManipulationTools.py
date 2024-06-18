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
def drawHelpPoints(output_Image, points):
    for point in points:
        #print("point: ", point)
        # Ensure the point is a flat list or tuple with two elements
        if isinstance(point, (list, tuple)) and len(point) == 2:
            # Convert the point coordinates to integers
            center = (int(point[0]), int(point[1]))
            cv.circle(output_Image, center, 5, (0, 0, 255), -1)
        else:
            print(f"Unexpected point format: {point}")
    return output_Image