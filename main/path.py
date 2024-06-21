import math
import numpy as np
import sys
import os
import time
import cv2 as cv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ComputerVision



def find_contour_center(contour):
    M = cv.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])  # Calculate the X coordinate of the centroid
        cY = int(M["m01"] / M["m00"])  # Calculate the Y coordinate of the centroid
        return (cX, cY)
    return None

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_angle(robot_position, ball_position, robot_orientation):
    print("here2")
    print("robot_position",robot_position)
    print("ball_position",ball_position)
    print("robot_orientation",robot_orientation)

    dy = ball_position[1] - robot_position[1]
    dx = ball_position[0] - robot_position[0]
    angle_to_target_radians = math.atan2(dy, dx)
    angle_to_target_degrees = math.degrees(angle_to_target_radians)
    angle = robot_orientation + angle_to_target_degrees
    print("here5")
    if abs(angle) > 180:
        angle = abs(angle) - 360
    return -angle

def find_close_ball(robot_position, balls, robot_orientation):
    closest_ball = None
    min_distance = float('inf')
    angle_to_turn = 0
    for ball in balls:
        
        distance = calculate_distance(robot_position, ball)
       
        if distance < min_distance:
            
            min_distance = distance
            closest_ball = ball
            
            angle_to_turn = calculate_angle(robot_position, ball, robot_orientation)
            
    return closest_ball, min_distance, angle_to_turn

def find_shortest_path(robot_position, robot_orientation, paired_help_points_and_balls, contours, drive_points):
    # If no help points are available, use drive points as a fallback
    if not paired_help_points_and_balls:
        print("No help point and ball pairs available.")
        return find_close_ball(robot_position, drive_points, robot_orientation)
    

    closest_help_point = None
    min_distance = float('inf')
    best_angle_to_turn = None
    selected_ball = None

    # Iterate over each help point and ball pair
    for help_point in paired_help_points_and_balls:
        if is_path_clear(robot_position, help_point.con, contours):
            distance = calculate_distance(robot_position, help_point.con)
            if distance < min_distance:
                min_distance = distance
                closest_help_point = help_point.con
                
                selected_ball = ComputerVision.ImageProcessor.find_contour_center(help_point.ball.con)
                best_angle_to_turn = calculate_angle(robot_position, help_point.con, robot_orientation)

    # If a help point is selected and it has a clear path to its associated ball
    if closest_help_point and selected_ball:
        return closest_help_point, selected_ball,best_angle_to_turn, min_distance
        

    # If no accessible help point is found, find the nearest drive point
    if not closest_help_point:
        print("No accessible help point found, looking for the nearest drive point.")
        closest_drive_point, drive_point_distance, drive_angle_to_turn = find_close_ball(robot_position, drive_points, robot_orientation)
       
       

        if calculate_distance(robot_position, closest_drive_point) < 50:
            print("before pop",drive_points)
            drive_points.remove(closest_drive_point)
            print("after pop",drive_points)
            time.sleep(30)
            closest_drive_point, drive_point_distance, drive_angle_to_turn = find_close_ball(robot_position, drive_points, robot_orientation)
           
           
        if closest_drive_point:
            return closest_drive_point,None ,drive_angle_to_turn,drive_point_distance
            
        else:
            print("No accessible drive points found.")
            return None, None,None,None

    return None


def create_parallel_lines(p1, p2, n, spacing):
    # Calculate the direction vector from p1 to p2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    # Normalize the direction vector
    length = math.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    # Calculate perpendicular vector to the direction vector
    perp_dx, perp_dy = -dy, dx

    lines = []
    # Offset to center the lines around the original line
    center_offset = -(n-1) / 2.0 * spacing
    
    # Generate n parallel lines
    for i in range(n):
        offset = center_offset + i * spacing
        # Calculate start and end points of each line, offset by the perpendicular vector
        line_start = (p1[0] + perp_dx * offset, p1[1] + perp_dy * offset)
        line_end = (p2[0] + perp_dx * offset, p2[1] + perp_dy * offset)
        lines.append((line_start, line_end))
    
    return lines

def is_path_clear(p1, p2, contours):
    p1 = (int(round(p1[0])), int(round(p1[1])))
    p2 = (int(round(p2[0])), int(round(p2[1])))
    # Generate parallel lines as polygons
    parallel_lines = create_parallel_lines(p1, p2, 10, 10)
    # Check each line for intersections with any of the contours
    for line in parallel_lines:
        line_contour = np.array([[[int(line[0][0]), int(line[0][1])]], [[int(line[1][0]), int(line[1][1])]]], dtype=np.int32)
        for contour in contours:
            if contour_intersect(contour, line_contour):
                return False
    return True
def contour_intersect(cnt_ref, cnt_query):
    # Adjust this function to handle a single line passed as cnt_query correctly
    for ref_idx in range(len(cnt_ref) - 1):
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx + 1][0]
        # Assume cnt_query is a single line segment
        C = cnt_query[0][0]
        D = cnt_query[1][0]
        if segment_intersects(A, B, C, D):
            return True
    return False

def segment_intersects(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
