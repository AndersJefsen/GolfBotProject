#Victor Justesen
import math
from collections import deque

def calculate_distance(p1, p2):
        '''
        print("calculate distance")
        print(f"p1: {p1}, type: {type(p1)}")
        print(f"p2: {p2}, type: {type(p2)}")
        '''
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(robot_position, ball_position, robot_orientation):
          
            dy = ball_position[1] - robot_position[1]
            dx = ball_position[0] - robot_position[0]
            angle_to_target_radians = math.atan2(dy, dx)
            angle_to_target_degrees = math.degrees(angle_to_target_radians)
            #print("angle to target",angle_to_target_degrees)
            # Adjust from East (atan2 default) to North
            angle=robot_orientation+angle_to_target_degrees
            if abs(angle) > 180:
                angle=abs(angle)-360
                        
            # Calculate relative angle considering current robot orientation
        
            
            return -angle

def find_close_ball(robot_position, balls,robot_orientation):
   
   
    
    closest_ball = None
    min_distance = float('inf')
    angle_to_turn = 0
    distance = None
    for ball in balls:
        distance = calculate_distance(robot_position, ball)
        if distance < min_distance:
            min_distance = distance
            closest_ball = ball
            angle_to_turn = calculate_angle(robot_position, ball,robot_orientation)
    
    return  closest_ball,distance, angle_to_turn

# BFS to find the shortest route from the robot's current position to any ball via help points
def bfs_path(start, goals, help_points, obstacles):
    all_points = set(help_points + [start] + goals)
    graph = {point: [] for point in all_points}
    print("add nodes to graph")
    # Connect nodes within graph, checking if the path is clear
    for point1 in all_points:
        for point2 in all_points:
            if point1 != point2 and (point1 in help_points and point2 in help_points or point1 == start or point2 in goals):
                if is_path_clear(point1, point2, obstacles):

                    graph[point1].append(point2)
    print("finding shortest path")
    # BFS to find the shortest path
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()
        if current in goals:
            
            return path
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    print ("no paths found")
    return None  # In case no path is found

def route_to_closest_ball(robot_position, balls, help_points, contours):
    print("making graph")
    paths = []
    for ball in balls:
        path = bfs_path(robot_position, [ball], help_points, contours)
        if path:
            paths.append((path, calculate_distance(robot_position, ball)))

    # Find the path to the closest ball
    if paths:
        print("found smallest path")
        return min(paths, key=lambda x: x[1])[0]  # Return only the path
    else:
        return None
    
import cv2
import numpy as np

def is_path_clear(p1, p2, contours):
    print("checking if path is clear")

    
    """
    Check if a line segment between points p1 and p2 intersects with any contour.

    Args:
    p1 (tuple): The starting point of the line segment.
    p2 (tuple): The ending point of the line segment.
    contours (list of np.ndarray): List of contours, where each contour is represented as an array of points.

    Returns:
    bool: True if the line intersects any contour, False otherwise.
    """
    p1 = (int(round(p1[0])), int(round(p1[1])))
    p2 =    (int(round(p2[0])), int(round(p2[1])))

    print("p1",p1)
    print("p2",p2)



    # Create a blank image that can fit all points
    max_x = max(p1[0], p2[0], max(contour[:, :, 0].max() for contour in contours))
    max_y = max(p1[1], p2[1], max(contour[:, :, 1].max() for contour in contours))
    img = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)

    # Draw the line on the image
    cv2.line(img, p1, p2, 255, 1)

    # Check each contour to see if it intersects the line
    for contour in contours:
        # Draw the contour in a new image
        contour_img = np.zeros_like(img)
        cv2.drawContours(contour_img, [contour], -1, 255, -1)  # Filled contour

        # Check for intersection
        intersection = cv2.bitwise_and(img, contour_img)
        if np.any(intersection > 0):
            return True

    return False    

# Example usage:






      
      