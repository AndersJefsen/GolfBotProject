import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.data import *

class TestData(unittest.TestCase):
    def setUp(self):
        self.data = Data()

    def create_mock_contour(self, x, y, w, h):
        return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
    
    #this test that ball is added if it is not detected before
    def test_add_new_balls(self):
        contours = [self.create_mock_contour(10, 10, 5, 5)]
        coordinates = [(12, 12)]
        self.data.addBalls(contours, coordinates)
        self.assertEqual(len(self.data.whiteballs), 1)
        self.assertEqual(self.data.whiteballs[0].cord, coordinates[0])
        self.assertEqual(self.data.whiteballs[0].det, 1)

    #this test that ball is not added if it is already detected
    def test_increment_detection(self):
        contours = [self.create_mock_contour(10, 10, 5, 5)]
        coordinates = [(12, 12)]
        self.data.addBalls(contours, coordinates)
        self.data.addBalls(contours, coordinates)
        self.assertEqual(len(self.data.whiteballs), 1)
        self.assertEqual(self.data.whiteballs[0].cord, coordinates[0])
        self.assertEqual(self.data.whiteballs[0].det, 2)

    #this test that ball is removed if it is not detected for 3 frames
    def test_remove_old_balls(self):
        contours = [self.create_mock_contour(10, 10, 5, 5)]
        coordinates = [(12, 12)]
        self.data.addBalls(contours, coordinates)
        self.data.whiteballs[0].recentlyDetected = False
        self.data.addBalls([], [])
        self.data.addBalls([], [])

        self.assertEqual(len(self.data.whiteballs), 0)

#this test that new ball is removed if it is not detected for 3 frames, but existing ball is not removed
    def test_mixed_ball_handling(self):
            # New ball detection
            contours_new = [self.create_mock_contour(10, 10, 5, 5)]
            coordinates_new = [(12, 12)]
            
            # Existing ball detection
            contours_existing = [self.create_mock_contour(20, 20, 5, 5)]
            coordinates_existing = [(22, 22)]

            # Add both balls
            self.data.addBalls(contours_existing+contours_new, coordinates_existing+coordinates_new)
            
            # Check that both balls are in the list
            self.assertEqual(len(self.data.whiteballs), 2)
            self.assertEqual(self.data.whiteballs[0].cord, (22, 22))
            self.assertEqual(self.data.whiteballs[0].det, 1)
            self.assertEqual(self.data.whiteballs[1].cord, (12, 12))
            self.assertEqual(self.data.whiteballs[1].det, 1)
            
            # Manually set recentlyDetected flags
            self.data.whiteballs[0].recentlyDetected = True
            self.data.whiteballs[1].recentlyDetected = False

            # New ball not detected, two frames with no balls detected
            self.data.addBalls([], [])
            self.data.addBalls([], [])
            
            # Check that the new ball has been removed and existing ball's det is decremented
            self.assertEqual(len(self.data.whiteballs), 1)
            self.assertEqual(self.data.whiteballs[0].cord, (22, 22))
            self.assertEqual(self.data.whiteballs[0].det, 0)


         
if __name__ == '__main__':
    unittest.main()
    


