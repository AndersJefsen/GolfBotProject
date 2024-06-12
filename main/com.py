import socket
from path import find_close_ball

# Define server address and port
SERVER_ADDRESS = '172.20.10.3'  #IP address of EV3
SERVER_PORT = 1024  # port server script


def connect_to_robot():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_ADDRESS, SERVER_PORT))
    return s
def close_connection(s):
    s.close()
def send_command(command,socket):
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

        

def command_robot(robot_position, balls, robot_orientation,socket):
        #ImageProcessor.get_robot_position()
        balls = balls #ImageProcessor.get_robot_position().get_balls()
       #ImageProcessor.get_robot_orientation()
        
       
       
        closest_ball, distance_to_ball, angle_to_turn = find_close_ball(robot_position, balls, robot_orientation)
        print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")
    
        print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
        # Get command input from the user
        command = f"TURN {angle_to_turn}"
        res = send_command(command,socket=socket)
        command = f"MOVE {distance_to_ball}"
        res = send_command(command,socket=socket)