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
def drawContours(output_image,contours):
    cv.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    return output_image