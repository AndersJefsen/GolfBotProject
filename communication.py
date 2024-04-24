import socket

# Server settings
#the third number needs to be changed each time the hotspot changes
HOST = '192.168.123.243'  # The IP address of your EV3 brick
PORT = 1024  # The same port as used by the server

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(command.encode('utf-8'))
    except ConnectionRefusedError:
       print("Could not connect to the server. Please check if the server is running and reachable.")
    except Exception as e:
        print(f"An error occurred: {e}")
# Example usage
    while True:
        # Here you need to implement a way to get the current position of the robot
        # For this example, we're setting a fixed position
        robot_position = (0, 0)  # This should be dynamically determined

        # And a way to dynamically get the positions of the balls
        balls = [(3, 4), (1, 2), (5, 5)]  # This should also be dynamically determined

        closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls)

        # Decision making for sending commands
        if distance_to_ball < 0.5:  # Assuming this is the distance threshold for "close enough to collect"
            send_command("COLLECT")
        elif abs(angle_to_turn) > 10:  # Assuming this is the angle threshold to decide if we need to turn
            if angle_to_turn > 0:
                send_command("RIGHT")
            else:
                send_command("LEFT")
        else:
            send_command("FORWARD")
        
        time.sleep(1) 
