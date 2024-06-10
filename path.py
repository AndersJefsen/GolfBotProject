import math

def find_closest_ball(robot_position, balls,robot_orientation):
   
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_angle(robot_position, ball_position,robot_orientation):
        dy = ball_position[1] - robot_position[1]
        dx = ball_position[0] - robot_position[0]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        relative_angle = angle_degrees - robot_orientation
        normalized_angle = (relative_angle + 180) % 360 - 180
        return normalized_angle
    
    closest_ball = None
    min_distance = float('inf')
    angle_to_turn = 0
    
    for ball in balls:
        distance = calculate_distance(robot_position, ball)
        if distance < min_distance:
            min_distance = distance
            closest_ball = ball
            angle_to_turn = calculate_angle(robot_position, ball,robot_orientation)
    
    return closest_ball, min_distance, angle_to_turn

# Example usage:
robot_position = (0, 0)  # Example robot position
balls = [(3, 4), (1, 2), (5, 5)]  # Example balls positions
robot_orientation=0
closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls, robot_orientation)

