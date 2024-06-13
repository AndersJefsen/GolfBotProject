import socket
from main.path import find_closest_ball

# Define server address and port
SERVER_ADDRESS = '172.20.10.3'  #IP address of EV3
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

if __name__ == "__main__":
        robot_position = (43,67)#ImageProcessor.get_robot_position()
        balls = [(0,0)] #ImageProcessor.get_robot_position().get_balls()
        robot_orientation =(91+90)%360  #ImageProcessor.get_robot_orientation()
        closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls, robot_orientation)
        print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")
    
        print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
        # Get command input from the user
        command = f"TURN {angle_to_turn}"
        send_command(command)
        command = f"FORWARD {distance_to_ball}"
        send_command(command)