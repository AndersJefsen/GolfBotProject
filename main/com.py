import socket
from path import find_close_ball
import asyncio
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

async def send_command_async(command, socket):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, send_command, command, socket)

        
async def command_robot_async(robot_position, balls, robot_orientation, socket, completion_flag):
    logger.info("Entering command_robot_async") 
    closest_ball, distance_to_ball, angle_to_turn = find_close_ball(robot_position, balls, robot_orientation)
    print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")
    
    print(f"TURN {angle_to_turn}", f"FORWARD {distance_to_ball}")
    
    command_turn = f"TURN {angle_to_turn}"
    command_move = f"MOVE {distance_to_ball}"
    
    res_turn = await send_command_async(command_turn, socket)
    res_move = await send_command_async(command_move, socket)
    
    # Set the completion flag to True when both commands are done
    completion_flag.set()
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
        return True

    