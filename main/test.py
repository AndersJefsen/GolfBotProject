import cv2 as cv
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main.ComputerVision import ComputerVision


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
    cv.imshow('inside_region',inside_region)
    cv.waitKey(0)
    # Isolate the outside region
    outside_region = cv.bitwise_and(black_background, black_background, mask=inverted_mask)
    cv.imshow('outside_region',outside_region)
    cv.waitKey(0)
    # Combine inside and outside regions
    final_image = cv.add(inside_region, outside_region)
    cv.imshow('final_image',final_image)
    cv.waitKey(0)
    return mask


def main():
    screenshot = cv.imread('testpic.jpg')
    outPutImage = screenshot.copy()
    arenaCorners = []
   
    findArena, outPutImage,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ComputerVision.find_arena(screenshot, outPutImage)
    if findArena:
        arenaCorners.append(bottom_left_corner)
        arenaCorners.append(bottom_right_corner)
        arenaCorners.append(top_left_corner)
        arenaCorners.append(top_right_corner)
     
    mask = createMask(screenshot,findArena)


