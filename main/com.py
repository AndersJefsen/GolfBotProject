import socket
from path import find_close_ball, calculate_angle, calculate_distance
import math

# Define server address and port
SERVER_ADDRESS = '172.20.10.14'  #IP address of EV3
SERVER_PORT = 1024  # port server script


def connect_to_robot():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_ADDRESS, SERVER_PORT))
    return s


def close_connection(s):
    s.close()


def send_command(command, socket):
    try:
        # Create a socket object

        # Send the command
        socket.sendall(command.encode('utf-8'))
        # Receive the response from the server
        response = socket.recv(1024).decode('utf-8')
        print(f"Response from server: {response}")
        return response
    except Exception as e:
        print(f"An error occurred: {e}")


def command_robot(robot_position, balls, robot_orientation, socket):
    #ImageProcessor.get_robot_position()
    balls = balls  #ImageProcessor.get_robot_position().get_balls()
    #ImageProcessor.get_robot_orientation()

    closest_ball, distance_to_ball, angle_to_turn = find_close_ball(robot_position, balls, robot_orientation)
    print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")

    print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
    # Get command input from the user
    command = f"TURN {angle_to_turn}"
    res = send_command(command, socket=socket)
    command = f"MOVE {distance_to_ball}"
    res = send_command(command, socket=socket)


def drive_robot_to_point(point, pos, socket):
    angle_to_turn = calculate_angle(pos, point)

    command = f"TURN {angle_to_turn}"
    res = send_command(command, socket=socket)

    distance_to_move = calculate_distance(pos, point)
    command = f"MOVE {distance_to_move}"
    res = send_command(command, socket=socket)

    # Assuming res indicates success or failure of the command
    # Modify this part according to your actual response handling
    if res == "success":  # Adjust according to your response logic
        return True
    else:
        return False



    #Maybe a returned boolean incase robot has arrived at destination?


def calculate_angle(pos, point):
    dx = point[0] - pos[0]
    dy = point[1] - pos[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def turn_robot(oriention, desired_angle):
    global socket  # Access the global socket variable

    oriention = oriention % 360
    desired_angle = desired_angle % 360

    diff = abs(oriention - desired_angle)
    if diff > 180:
        diff = 360 - diff

    command = f"TURN {diff}"
    res = send_command(command, socket=socket)
    return diff == 0



def release():
    command = f"RELEASE"
    res = send_command(command, socket=socket)
    return res


def move_to_position_and_release(point, pos, orientation, socket):
    # Drive the robot to the target point
    arrived = drive_robot_to_point(point, pos, socket)
    if not arrived:
        return False

    #Target is 180, its the orientaion of the small goal.
    correct_angle = turn_robot(orientation, 180)
    if not correct_angle:
        return False

    #løslad lortet.
    release_res = release(socket)
    return release_res


# Example usage
# Assume socket is already created and connected to the robot
# point = (x_target, y_target)
# pos = (x_current, y_current)
# orientation = current_orientation_in_degrees
# socket = your_socket_instance

#-------------Hvordan kalder jeg på lortet?
#result = move_to_position_and_release(point, pos, orientation, socket)
