import sys
import math
import unittest
import os

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.path import calculate_distance, calculate_angle, find_close_ball,create_parallel_lines, is_path_clear

class TestPath(unittest.TestCase):
    def test_calculate_distance(self):
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
        robot_position = (50, 105)
        ball_position = (75, 50)
        robot_orientation = 180
        expected_angle =-114.44
        self.assertAlmostEqual(calculate_angle(robot_position, ball_position, robot_orientation), expected_angle, places=2)
        
        robot_position = (0, 0)
        ball_position = (1, 1)
        robot_orientation = 270
        expected_angle = 45
        self.assertEqual(calculate_angle(robot_position, ball_position, robot_orientation), expected_angle)


    def test_find_close_ball(self):
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
            self.assertAlmostEqual(distance, expected_distance,places=2)
            self.assertEqual(angle, expected_angle)

 

    def test_create_parallel_lines(self):
        p1 = (0, 0)
        p2 = (10, 0)
        n = 3
        spacing = 1
        expected_lines = [((0, -1), (10, -1)), ((0, 0), (10, 0)), ((0, 1), (10, 1))]
        self.assertEqual(create_parallel_lines(p1, p2, n, spacing), expected_lines)

    def test_is_path_clear(self):
            p1 = (0, 0)
            p2 = (10, 10)
            contours = [
                np.array([[[5, 5]], [[6, 6]]], dtype=np.int32),
                np.array([[[3, 3]], [[4, 4]]], dtype=np.int32)
            ]
            self.assertTrue(is_path_clear(p1, p2, contours))
         
            p1 = (0, 0)
            p2 = (10, 10)
            # Define contours that will intersect with the parallel lines
            contours = [
                np.array([[[5, 5]], [[6, 6]]], dtype=np.int32),
                np.array([[[7, 7]], [[8, 8]]], dtype=np.int32),
                np.array([[[9, 9]], [[10, 10]]], dtype=np.int32),
                np.array([[[11, 11]], [[12, 12]]], dtype=np.int32),
                np.array([[[13, 13]], [[14, 14]]], dtype=np.int32)
            ]
            # Assert that is_path_clear returns False due to intersections
            self.assertFalse(is_path_clear(p1, p2, contours))

if __name__ == '__main__':
    unittest.main()

def test_is_path_clear_false(self):
    p1 = (0, 0)
    p2 = (10, 10)
    contours = [
        np.array([[[5, 5]], [[6, 6]]], dtype=np.int32),
        np.array([[[7, 7]], [[8, 8]]], dtype=np.int32),
        np.array([[[9, 9]], [[10, 10]]], dtype=np.int32),
        np.array([[[11, 11]], [[12, 12]]], dtype=np.int32),
        np.array([[[13, 13]], [[14, 14]]], dtype=np.int32)
    ]
    self.assertFalse(is_path_clear(p1, p2, contours))