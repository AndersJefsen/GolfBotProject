import socket

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
    while True:
        # Get command input from the user
        command = input("Enter command (e.g., FORWARD 10, TURN 90, COLLECT, RELEASE, BACKWARD 10, TURN -90 ): ")
        if command.lower() == 'exit':
            break
        send_command(command)
