from windowcapture import WindowCapture
from time import time
import cv2 as cv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
from ultralytics import YOLO
import numpy as np
from ComputerVision import ComputerVision

def createMask(imageToDetectOn,points):
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros(imageToDetectOn.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [points], 255)

    # Invert the mask
    inverted_mask = cv.bitwise_not(mask)

    # Create black background for outside region
    black_background = np.zeros_like(imageToDetectOn)

    # Isolate the inside region
    inside_region = cv.bitwise_and(imageToDetectOn, imageToDetectOn, mask=mask)

    # Isolate the outside region
    outside_region = cv.bitwise_and(black_background, black_background, mask=inverted_mask)

    # Combine inside and outside regions
    final_image = cv.add(inside_region, outside_region)

    return mask


def main():
    
