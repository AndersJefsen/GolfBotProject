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
        cls.orange_ball_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.white_ball_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.egg_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.arena_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.cross_image = cv2.imread('images/Blandet/outerball.jpg')
        cls.robot_image = cv2.imread('images/Blandet/outerball.jpg')


    def test_find_balls_hsv(self):
        """
        TC001:
        Test case for the find_balls_hsv method of the ImageProcessor class.
        It checks if the method returns non-empty contours of balls of a picture.

        """
        contours = ComputerVision.ImageProcessor.find_balls_hsv(self.white_ball_image, 50, 400)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_orangeball_hsv(self):
        """
        TC002:
        Test case for the find_orangeball_hsv method of the ImageProcessor class.

        This method tests the functionality of the find_orangeball_hsv method by passing an image containing orange ball
        and checking if the returned contours of orange balls are not None and have a length greater than 0.

        """
        contours = ComputerVision.ImageProcessor.find_orangeball_hsv(self.orange_ball_image, 50, 700)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_bigball_hsv(self):
        """
        TC003:
        Test case for the find_bigball_hsv method in the ImageProcessor class.

        This test verifies that the find_bigball_hsv method returns valid contours for a given image.

        """
        contours = ComputerVision.ImageProcessor.find_bigball_hsv(self.egg_image, 500, 10000, 0.7, 1.3)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

   
    def test_find_arena(self):
        """
        TC004:
        Test case for the find_Arena method in the ImageProcessor class.
        
        This method tests whether the find_Arena method correctly identifies the arena in an image.
        It calls the find_Arena method with the arena image and a copy of the arena image, and checks
        if the method returns True, indicating that the arena was found.
        """
        found, _, _, _, _, _, _ = ComputerVision.ImageProcessor.find_Arena(self.arena_image, self.arena_image.copy())
        self.assertTrue(found)

    def test_find_cross_contours(self):
        """
        TC005:
        Test case for the find_cross_contours method of the ImageProcessor class.
        It verifies that the method returns non-empty contours.

        """
        contours = ComputerVision.ImageProcessor.find_cross_contours(self.cross_image)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

    def test_find_robot(self):
        """
        TC006:
        
        Test case for the find_robot method of the ImageProcessor class.

        This test verifies that the find_robot method correctly detects the contours of a robot in an image.

        It checks that the returned contours are not None and that the number of contours is greater than zero.

        """
        contours = ComputerVision.ImageProcessor.find_robot(self.robot_image, 10, 100000)
        self.assertIsNotNone(contours)
        self.assertGreater(len(contours), 0)

if __name__ == '__main__':
    unittest.main()
