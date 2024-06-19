from data import Data as Data
import imageManipulationTools
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision
def findArena_flow(screenshot,output_image,data:Data):
    findArena, output_image,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, filtered_contoures = ComputerVision.ImageProcessor.find_Arena(screenshot, output_image)
    print("her",findArena,bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner)
    if findArena:
        arenaCorners = []
        arenaCorners.append(bottom_left_corner)
        arenaCorners.append(bottom_right_corner)

        arenaCorners.append(top_right_corner)
        arenaCorners.append(top_left_corner)
        data.arenaCorners = arenaCorners
        data.mask = imageManipulationTools.createMask(screenshot,arenaCorners)
        print("mask Created")
        
    return findArena
