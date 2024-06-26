import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.data import *

class TestData(unittest.TestCase):
    """
    A test case for the Data class.

    This test case includes tests for adding, incrementing, and removing balls in the Data class.
    """

    def setUp(self):
        self.data = Data()

    def create_mock_contour(self, x, y, w, h):
        return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
    
    #this test that ball is added if it is not detected before
    def test_add_new_balls(self):
        """
        TC007:
        Test case for the addBalls method of the Data class.

        This test verifies that the addBalls method correctly adds new balls to the data object.

        Steps:
        1. Create a mock contour with specific coordinates and dimensions.
        2. Create a list of contours containing the mock contour.
        3. Create a list of coordinates containing a single coordinate.
        4. Call the addBalls method with the contours and coordinates.
        5. Assert that the number of white balls in the data object is 1.
        6. Assert that the coordinates of the first white ball match the provided coordinate.
        7. Assert that the detection count of the first white ball is 1.
        """
        contours = [self.create_mock_contour(10, 10, 5, 5)]
        coordinates = [(12, 12)]
        self.data.addBalls(contours, coordinates)
        self.assertEqual(len(self.data.whiteballs), 1)
        self.assertEqual(self.data.whiteballs[0].cord, coordinates[0])
        self.assertEqual(self.data.whiteballs[0].det, 1)

    #this test that ball is not added if it is already detected
    def test_increment_detection(self):
        """
        TC008:
        Test case for the increment detection functionality.

        This test verifies that the `addBalls` method correctly increments the detection count
        for white balls when called multiple times with the same contours and coordinates.

        Steps:
        1. Create a mock contour with specific dimensions.
        2. Create a list of contours containing the mock contour.
        3. Create a list of coordinates containing a single coordinate.
        4. Call the `addBalls` method twice with the same contours and coordinates.
        5. Assert that the length of `whiteballs` is 1.
        6. Assert that the `cord` attribute of the first white ball is equal to the first coordinate.
        7. Assert that the `det` attribute of the first white ball is equal to 2.

        """
        contours = [self.create_mock_contour(10, 10, 5, 5)]
        coordinates = [(12, 12)]
        self.data.addBalls(contours, coordinates)
        self.data.addBalls(contours, coordinates)
        self.assertEqual(len(self.data.whiteballs), 1)
        self.assertEqual(self.data.whiteballs[0].cord, coordinates[0])
        self.assertEqual(self.data.whiteballs[0].det, 2)

    #this test that ball is removed if it is not detected for 3 frames
    def test_remove_old_balls(self):
        """
        TC009:
        Test case for the remove_old_balls method.

        This test case verifies that the remove_old_balls method correctly removes old balls from the data object.

        Steps:
        1. Create a mock contour and coordinates.
        2. Add balls to the data object using the addBalls method.
        3. Set the recentlyDetected attribute of the first white ball to False.
        4. Add two empty lists of balls to the data object using the addBalls method.
        5. Verify that the length of the whiteballs list in the data object is 0.

        """
        contours = [self.create_mock_contour(10, 10, 5, 5)]
        coordinates = [(12, 12)]
        self.data.addBalls(contours, coordinates)
        self.data.whiteballs[0].recentlyDetected = False
        self.data.addBalls([], [])
        self.data.addBalls([], [])

        self.assertEqual(len(self.data.whiteballs), 0)

    #this test that new ball is removed if it is not detected for 3 frames, but existing ball is not removed
    def test_mixed_ball_handling(self):
        """
        TC010:
        Test case for handling mixed ball detection.

        This test case verifies the behavior of the `addBalls` method when both new and existing balls are detected.
        It checks that the balls are correctly added to the list, their properties are set correctly, and the recentlyDetected
        flag is updated accordingly. It also tests the removal of a ball and the decrementing of the det property.

        Steps:
        1. Create mock contours and coordinates for new and existing balls.
        2. Add both balls to the data object using the `addBalls` method.
        3. Verify that both balls are in the list and their properties are set correctly.
        4. Manually set the recentlyDetected flags for the balls.
        5. Add two frames with no balls detected.
        6. Verify that the new ball has been removed and the existing ball's det property is decremented.

        """
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
    


