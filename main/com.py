import socket
import math
import asyncio
import logging
from path import calculate_distance, calculate_angle, find_close_ball

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = '172.20.10.3'  # IP address of EV3
SERVER_PORT = 1024  # Port server script



def connect_to_robot(server_address =SERVER_ADDRESS, server_port = SERVER_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server_address, server_port))
    return s

def close_connection(s):
    s.close()

def send_command(command, sock):
    try:
        sock.sendall(command.encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        print(f"Response from server: {response}")
        return response
    except Exception as e:
        print(f"An error occurred: {e}")

async def send_command_async(command, sock):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, send_command, command, sock)

async def command_robot_async(robot_position, balls, robot_orientation, sock, completion_flag):
    logger.info("Entering command_robot_async")
    closest_ball, distance_to_ball, angle_to_turn = find_close_ball(robot_position, balls, robot_orientation)
    print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")

    print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")

    command_turn = f"TURN {angle_to_turn}"
    command_move = f"MOVE {distance_to_ball}"

    res_turn = await send_command_async(command_turn, sock)
    res_move = await send_command_async(command_move, sock)

    completion_flag.set()

def command_robot(robot_position, balls, robot_orientation, sock):
    closest_ball, distance_to_ball, angle_to_turn = find_close_ball(robot_position, balls, robot_orientation)
    print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")

    print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
    command = f"TURN {angle_to_turn}"
    res = send_command(command, sock)
    command = f"MOVE {distance_to_ball}"
    res = send_command(command, sock)

def drive_robot_to_point(point, pos, orientation, sock):
    angle_to_turn = calculate_angle(pos, point, orientation)
    distance_to_move = calculate_distance(pos, point)

    # Print the calculated distance and angle
    print(f"Calculated angle to turn: {angle_to_turn} degrees")
    print(f"Calculated distance to move: {distance_to_move} units")

    command = f"TURN {angle_to_turn}"
    res = send_command(command, sock)

    command = f"MOVE_GOAL {distance_to_move}"
    res = send_command(command, sock)

    if res == "success":
        return True
    else:
        return False


def turn_robot(orientation, desired_angle, sock):
    orientation = orientation % 360
    desired_angle = desired_angle % 360

    diff = abs(orientation - desired_angle)
    if diff > 180:
        diff = 360 - diff

    command = f"TURN {diff}"
    res = send_command(command, sock)
    print(diff)
    return diff == 0

def release(sock):
    command = f"RELEASE"
    res = send_command(command, sock)
    return res

def move_to_position_and_release(point, pos, orientation, sock):
    arrived = drive_robot_to_point(point, pos, orientation, sock)
    if not arrived:
        return False

    correct_angle = turn_robot(orientation, 180, sock)
    if not correct_angle:
        return False

    release_res = release(sock)
    return release_res

# Example usage
# point = (x_target, y_target)
# pos = (x_current, y_current)
# orientation = current_orientation_in_degrees

