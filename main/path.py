#Victor Justesen
import math

def find_close_ball(robot_position, balls,robot_orientation):
   
    def calculate_distance(p1, p2):
        print("calculate distance")
        print(f"p1: {p1}, type: {type(p1)}")
        print(f"p2: {p2}, type: {type(p2)}")
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_angle(robot_position, ball_position,robot_orientation):
        print("robot orientation:")
        dy = ball_position[1] - robot_position[1]
        dx = ball_position[0] - robot_position[0]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        relative_angle = -((angle_degrees - robot_orientation)%360)
        return relative_angle
    

    print("find closest ball")
    closest_ball = None
    min_distance = float('inf')
    angle_to_turn = 0
    print("balls:")
    print(balls)
    print("balls loop")
    for ball in balls:
        distance = calculate_distance(robot_position, ball)
        if distance < min_distance:
            min_distance = distance
            closest_ball = ball
            angle_to_turn = calculate_angle(robot_position, ball,robot_orientation)+180
    
    return  closest_ball,distance, angle_to_turn

# Example usage:
# robot_position = (0, 0)  # Example robot position
# balls = [(3, 4), (1, 2), (5, 5)]  # Example balls positions
# robot_orientation=0
# closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls, robot_orientation)
# print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}"