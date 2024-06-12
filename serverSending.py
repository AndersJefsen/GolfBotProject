import socket

# Server settings
#the third number needs to be changed each time the hotspot changes
HOST = '172.20.10.3'  # The IP address of your EV3 brick
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
 command = input()
 send_command(command)