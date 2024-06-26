import sys
import math
import unittest
import os

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.path import calculate_distance, calculate_angle, find_close_ball,create_parallel_lines, is_path_clear

class TestPath(unittest.TestCase):
    def test_calculate_distance(self):
        """
        TC011-TC013:
        Test case for the calculate_distance function.

        This test case checks the correctness of the calculate_distance function by comparing the calculated distance
        with the expected distance for different input points.

        Test cases:
        1. When both points are at (0, 0), the expected distance is 0.
        2. When p1 is at (1, 1) and p2 is at (4, 5), the expected distance is 5.
        3. When p1 is at (160, 120) and p2 is at (0, 0), the expected distance is 200.
        """
        p1 = (0, 0)
        p2 = (0, 0)
        expected_distance = 0
        self.assertEqual(calculate_distance(p1, p2), expected_distance)

        p1 = (1, 1)
        p2 = (4, 5)
        expected_distance = 5
        self.assertEqual(calculate_distance(p1, p2), expected_distance)

        p1 = (160, 120)
        p2 = (0, 0)
        expected_distance = 200
        self.assertAlmostEqual(calculate_distance(p1, p2), expected_distance)

#the robot orientation is at 0 degrees when pointing to positive x-axis
    def test_calculate_angle(self):
         """
         TC014-TC015:
         Test case for the calculate_angle function.

         This test verifies the correctness of the calculate_angle function by comparing the calculated angle
         with the expected angle for different robot positions, ball positions, and robot orientations.

         Test cases:
         1. Test with robot_position = (50, 105), ball_position = (75, 50), robot_orientation = 180.
             The expected angle is -114.44.
         2. Test with robot_position = (0, 0), ball_position = (1, 1), robot_orientation = 270.
             The expected angle is 45.
         """
         robot_position = (50, 105)
         ball_position = (75, 50)
         robot_orientation = 180
         expected_angle = -114.44
         self.assertAlmostEqual(calculate_angle(robot_position, ball_position, robot_orientation), expected_angle, places=2)

         robot_position = (0, 0)
         ball_position = (1, 1)
         robot_orientation = 270
         expected_angle = 45
         self.assertEqual(calculate_angle(robot_position, ball_position, robot_orientation), expected_angle)


    def test_find_close_ball(self):
        """
        TC016-TC017:
        Test case for the find_close_ball function.

        This test case verifies the correctness of the find_close_ball function by checking if it returns the expected
        ball, distance, and angle values for different robot positions, balls, and orientations.

        """
        robot_position = (0, 0)
        balls = [(1, 1), (2, 2), (3, 3)]
        robot_orientation = 0
        expected_ball = (1, 1)
        expected_distance = math.sqrt(2)
        expected_angle = -45
        ball, distance, angle = find_close_ball(robot_position, balls, robot_orientation)
        self.assertEqual(ball, expected_ball)
        self.assertAlmostEqual(distance, expected_distance)
        self.assertEqual(angle, expected_angle)

        robot_position = (10, 10)
        balls = [(5, 5), (8, 8), (12, 12)]
        robot_orientation = 90
        expected_ball = (8, 8)
        expected_distance = 2.828
        expected_angle = 45
        ball, distance, angle = find_close_ball(robot_position, balls, robot_orientation)
        self.assertEqual(ball, expected_ball)
        self.assertAlmostEqual(distance, expected_distance, places=2)
        self.assertEqual(angle, expected_angle)

 

    def test_create_parallel_lines(self):
        """
        TC018:
        Test case for the create_parallel_lines function.

        This test verifies that the create_parallel_lines function correctly creates parallel lines
        given two points, the number of lines, and the spacing between them.

        The expected result is a list of tuples representing the parallel lines.

        """
        p1 = (0, 0)
        p2 = (10, 0)
        n = 3
        spacing = 1
        expected_lines = [((0, -1), (10, -1)), ((0, 0), (10, 0)), ((0, 1), (10, 1))]
        self.assertEqual(create_parallel_lines(p1, p2, n, spacing), expected_lines)

    def test_is_path_clear(self):
        """
        TC019-TC020:
        Test case for the is_path_clear function.

        This test checks if the is_path_clear function correctly determines if the path between two points is clear
        based on the given contours.

        The test creates two points, p1 and p2, and a list of contours. It then calls the is_path_clear function
        with these inputs and asserts that the function returns True, indicating that the path is clear.

        Note: This test assumes that the is_path_clear function has been implemented correctly.

        """
        p1 = (0, 0)
        p2 = (0, 3)
        contours = [
            np.array([[[5, 5]], [[6, 6]]], dtype=np.int32),
            np.array([[[3, 3]], [[4, 4]]], dtype=np.int32)
        ]
        self.assertTrue(is_path_clear(p1, p2, contours))

        p1 = (1, 1)
        p2 = (1, 10)
        contours = [
            np.array([[[1, 2]], [[6, 6]]], dtype=np.int32),
            np.array([[[1, 8]], [[4, 4]]], dtype=np.int32)
        ]
        self.assertFalse(is_path_clear(p1, p2, contours))      
  
if __name__ == '__main__':
    unittest.main()
