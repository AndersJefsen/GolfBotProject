import socket
from path import find_closest_ball

# Define server address and port
SERVER_ADDRESS = '192.168.242.243'  #IP address of EV3
SERVER_PORT = 1024  # port server script

def send_command(command):
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect to the server
        s.connect((SERVER_ADDRESS, SERVER_PORT))
        # Send the command
        s.sendall(command.encode('utf-8'))
        # Close the connection
        s.close()
    except Exception as e:
        print(f"An error occurred: {e}")

def command_robot(robot_position, balls, robot_orientation):
        print(robot_orientation)
        robot_position = robot_position#ImageProcessor.get_robot_position()
        balls = balls #ImageProcessor.get_robot_position().get_balls()
        robot_orientation =(robot_orientation+90)%360  #ImageProcessor.get_robot_orientation()
        closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls, robot_orientation)
        print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")
    
        print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
        # Get command input from the user
        command = f"TURN {angle_to_turn}"
        send_command(command)
        command = f"FORWARD {distance_to_ball}"
        send_command(command)