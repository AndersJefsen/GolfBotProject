import math
from collections import deque
import numpy as np
import cv2

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_angle(robot_position, ball_position, robot_orientation):
    dy = ball_position[1] - robot_position[1]
    dx = ball_position[0] - robot_position[0]
    angle_to_target_radians = math.atan2(dy, dx)
    angle_to_target_degrees = math.degrees(angle_to_target_radians)
    angle = robot_orientation + angle_to_target_degrees
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

def bfs_path(start, goals, help_points, obstacles, show_visualization=True,imagee="he"):
    all_points = set(help_points + [start] + goals)
    graph = {point: [] for point in all_points}
    
    if show_visualization:
        img = imagee
    
    for point1 in all_points:
        for point2 in all_points:
            if point1 != point2 and is_path_clear(point1, point2, obstacles):
                graph[point1].append(point2)
                if show_visualization:
                    cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)  # Draw in green

    valid_paths = []
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()
        if current in goals:
            valid_paths.append(path)
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    if show_visualization:
        cv2.imshow("Valid Paths", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return valid_paths

def route_to_closest_ball(robot_position, balls, help_points, contours,image):
    print("making graph")
    paths = []
    for ball in balls:
        path = bfs_path(robot_position, [ball], help_points, contours, True,image)
        if path:
            paths.append((path, calculate_distance(robot_position, ball)))

    if paths:
        print("found smallest path")
        return min(paths, key=lambda x: x[1])[0]  # Return only the path
    else:
        return None

def create_path_polygon(p1, p2, width):
    # Vector from p1 to p2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    # Normalize the direction vector
    length = math.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    # Perpendicular vector to the direction vector
    px, py = -dy, dx
    # Calculate the offset for the width
    offset = width / 2
    # Create polygon vertices by offsetting the original line
    poly_points = np.array([
        [p1[0] + px * offset, p1[1] + py * offset],
        [p2[0] + px * offset, p2[1] + py * offset],
        [p2[0] - px * offset, p2[1] - py * offset],
        [p1[0] - px * offset, p1[1] - py * offset]
    ], dtype=np.int32)
    return poly_points

def is_path_clear(p1, p2, contours):
    p1 = (int(round(p1[0])), int(round(p1[1])))
    p2 = (int(round(p2[0])), int(round(p2[1])))
    path_contour = np.array([[[p1[0], p1[1]]], [[p2[0], p2[1]]]], dtype=np.int32)
    #path_contour =create_path_polygon(p1, p2, 5)
    for contour in contours:
        if contour_intersect(contour, path_contour):
            return False
    return True

def contour_intersect(cnt_ref, cnt_query):
    for ref_idx in range(len(cnt_ref) - 1):
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx + 1][0]
        for query_idx in range(len(cnt_query) - 1):
            C = cnt_query[query_idx][0]
            D = cnt_query[query_idx + 1][0]
            if segment_intersects(A, B, C, D):
                return True
    return False

def segment_intersects(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
