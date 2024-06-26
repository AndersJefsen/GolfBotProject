import os
import sys
import unittest
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision

class TestImageProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load sample images for testing
        cls.orange_ball_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.white_ball_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.egg_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.arena_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.cross_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.robot_image = cv2.imread('images/Blandet/outerball.jpg')


    def test_find_balls_hsv(self):
        contours = ComputerVision.ImageProcessor.find_balls_hsv(self.white_ball_image,50,400)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_orangeball_hsv(self):
        contours = ComputerVision.ImageProcessor.find_orangeball_hsv(self.orange_ball_image, 50, 700)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_bigball_hsv(self):
        contours = ComputerVision.ImageProcessor.find_bigball_hsv(self.egg_image,500,10000,0.7,1.3)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_arena(self):
        found, _, _, _, _, _, _ = ComputerVision.ImageProcessor.find_Arena(self.arena_image, self.arena_image.copy())
        self.assertTrue(found)

    def test_find_cross_contours(self):
        contours = ComputerVision.ImageProcessor.find_cross_contours(self.cross_image)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_robot(self):
        contours = ComputerVision.ImageProcessor.find_robot(self.robot_image, 10, 100000)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

if __name__ == '__main__':
    unittest.main()
