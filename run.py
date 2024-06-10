from path import find_closest_ball

def send_commands_via_ssh(host, port, username, password, commands):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port, username, password)
        
        for command in commands:
            stdin, stdout, stderr = client.exec_command(command)
            print(f"Output: {stdout.read().decode()}")
            err = stderr.read().decode()
            if err:
                print(f"Error: {err}")
                
    finally:
        client.close()

if __name__ == "__main__":
    # Example robot parameters
    robot_position = (0,0)#ImageProcessor.get_robot_position()
    balls = [(1,1)] #ImageProcessor.get_robot_position().get_balls()
    robot_orientation =0  #ImageProcessor.get_robot_orientation()

    # Call imported functions
    closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls, robot_orientation)
    print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball}, Angle to turn: {angle_to_turn}")


    # Define SSH connection details and commands based on previous results
    host = '172.20.10.3'
    port = 1024
    username = 'robot'
    password = 'maker'
    commands = [
        f"TURN {angle_to_turn}",
        f"FORWARD {distance_to_ball}"
    ]

    # Send commands via SSH
    #send_commands_via_ssh(host, port, username, password, commands)
    
    #tests, right from camera is robot orientation 0
test_cases = [
    {"robot_position": (0, 0), "balls": [(0, 1)], "robot_orientation": 0, "expected_angle": -90},
    {"robot_position": (0, 0), "balls": [(0, 1)], "robot_orientation": 180, "expected_angle": 90},
    {"robot_position": (0, 0), "balls": [(1, 1)], "robot_orientation": 0, "expected_angle": -45},
    {"robot_position": (0, 0), "balls": [(1, 1)], "robot_orientation": 180, "expected_angle": 135},
    {"robot_position": (0, 0), "balls": [(1, 0)], "robot_orientation": 0, "expected_angle": 0},
    {"robot_position": (0, 0), "balls": [(1, 0)], "robot_orientation": 180, "expected_angle": 180}
]

for case in test_cases:
    closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(
        case["robot_position"], case["balls"], case["robot_orientation"]
    )
    print(f"Test: Robot at {case['robot_position']} with orientation {case['robot_orientation']}°")
    print(f"Closest ball: {closest_ball}, Distance: {distance_to_ball:.2f}, Angle to turn: {angle_to_turn}°")
    print(f"Expected Angle to turn: {case['expected_angle']}°\n")