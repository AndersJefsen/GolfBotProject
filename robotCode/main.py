#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, TouchSensor, ColorSensor, InfraredSensor, UltrasonicSensor, GyroSensor
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import socket

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.

# Create your objects here.
ev3 = EV3Brick()
arm_motor = Motor(Port.A)
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
# Initialize the drive base.
robot = DriveBase(left_motor, right_motor, wheel_diameter=43.2, axle_track=143)
robot.settings(straight_speed=300, straight_acceleration=200, turn_rate=45, turn_acceleration=45)
# Initialize the gyro sensor.
gyro_sensor = GyroSensor(Port.S1)
ultrasonic_sensor = UltrasonicSensor(Port.S4)

# Reset the gyro sensor to zero.
gyro_sensor.reset_angle(0)

# Define a function to calibrate the gyro sensor
def calibrate_gyro(sensor):
    # Reset the gyro sensor to zero
    sensor.reset_angle(0)
    # Wait to stabilize
    wait(200)

    # Read angle multiple times, calculate the average
    angle_sum = 0
    for _ in range(100):
        angle_sum += sensor.angle()
        wait(10)

    # Calculate the average offset
    offset = angle_sum / 100

    # Set the gyro angle to the calculated offset
    sensor.reset_angle(offset)

# Function to collect.
def arm_collect(speed, time):
    arm_motor.run(600)
    wait(time)
    arm_motor.stop()

def arm_release(speed, time):
    arm_motor.run(-200)
    wait(time)
    arm_motor.stop()

def stop():
    right_motor.stop()
    left_motor.stop()


def turn_by_degrees(degrees):
    correction_factor_right =  0.99
    correction_factor_left = 1.12
    gyro_sensor.reset_angle(0)  # Reset the gyro sensor's angle
    target_angle = degrees


    # Turning the robot until the target angle is reached
    if target_angle < 0:
        while abs(gyro_sensor.angle())*correction_factor_left < abs(target_angle):
            print("gyro-sensor angle: ",abs(gyro_sensor.angle()))
            print("target angle: ", abs(target_angle) )
            robot.drive(0, -20)
            wait(100)  # Small delay
        robot.stop()  # Stop the robot once the target angle is reached


    else:
        while abs(gyro_sensor.angle())*correction_factor_right < abs(target_angle):
            print("gyro-sensor angle: ",abs(gyro_sensor.angle()))
            print("target angle: ", abs(target_angle) )
            robot.drive(0, 20)
            wait(100)
        robot.stop()  # Stop the robot once the target angle is reached




    gyro_sensor.reset_angle(0)
    robot.stop()

# Definer en funktion til at beregne den aktuelle rotation fra motorvinklerne
def get_current_rotations():
  
    

    left_rotations = abs(left_motor.angle() / 360)
    
    right_rotations = abs(right_motor.angle() / 360)

   
    return (left_rotations + right_rotations) / 2

def move_straight_goal(distance_cm):
    # Beregn antallet af rotationer, der kræves for at køre den ønskede distance
    wheel_diameter_cm = 4.32  # Diameteren på hjulet i cm
    wheel_circumference_cm = wheel_diameter_cm * 3.14159  # Omkredsen af hjulet (πd)

    # Antallet af rotationer der kræves for at køre den ønskede distance
    rotations_needed = abs(distance_cm / wheel_circumference_cm)

    # Reset motorerne for at starte fra nul
    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    correction_factor = 0.99  # Adjust this based on your robot's calibration
    speed = 200 if distance_cm >= 0 else -200
    arm_motor.run(600)
    # Kør motorerne direkte uden brug af robot.drive
    while abs(get_current_rotations()) < abs(rotations_needed):
        if ultrasonic_sensor.distance() < 180:
            left_motor.stop()
            right_motor.stop()
            arm_motor.stop()
            print("Obstacle detected. Stopping the robot.")
            wait(300)
            robot.drive(50, 0)
            wait(750)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_release(100,10000)            
            wait(50)
            break

        # Juster hastigheden for at korrigere forskelle mellem motorerne
        left_speed = speed * correction_factor
        right_speed = speed

        left_motor.run(left_speed)
        right_motor.run(right_speed)

        wait(10)  # Kort pause for at undgå for hyppige opdateringer

    # Stop motorerne når måldistancen er nået

    robot.stop()
    left_motor.stop()
    right_motor.stop()
    arm_motor.stop()
    print("Final distance:", get_current_rotations() * wheel_circumference_cm)

def move_straight(distance_cm):
    # Beregn antallet af rotationer, der kræves for at køre den ønskede distance
    wheel_diameter_cm = 4.32  # Diameteren på hjulet i cm
    wheel_circumference_cm = wheel_diameter_cm * 3.14159  # Omkredsen af hjulet (πd)

    # Antallet af rotationer der kræves for at køre den ønskede distance
    rotations_needed = abs(distance_cm) / wheel_circumference_cm
    # Reset motorerne for at starte fra nul
    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    correction_factor = 0.99  # Adjust this based on your robot's calibration
    speed = 300 if distance_cm >= 0 else -300
    arm_motor.run(600)
    # Kør motorerne direkte uden brug af robot.drive
    while abs(get_current_rotations()) < abs(rotations_needed):
        if ultrasonic_sensor.distance() < 180:
            left_motor.stop()
            right_motor.stop()
            arm_motor.stop()
            print("Obstacle detected. Stopping the robot.")
            # Kør baglæns lidt
            robot.drive(-1000, 0)
            wait(500)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            break
        print(" distance:", get_current_rotations() * wheel_circumference_cm)

        # Juster hastigheden for at korrigere forskelle mellem motorerne
        left_speed = speed * correction_factor
        right_speed = speed

        left_motor.run(left_speed)
        right_motor.run(right_speed)

        wait(10)  # Kort pause for at undgå for hyppige opdateringer
        print(" distance:", get_current_rotations() * wheel_circumference_cm)
    robot.stop()
    left_motor.stop()
    right_motor.stop()

    # Stop motorerne når måldistancen er nået
    arm_collect(600,3000)

    
    arm_motor.stop()
    print("Final distance:", get_current_rotations() * wheel_circumference_cm)

# Bruges til edge cases
def move_straight_collect(distance_cm):
    # Beregn antallet af rotationer, der kræves for at køre den ønskede distance
    wheel_diameter_cm = 4.32  # Diameteren på hjulet i cm
    wheel_circumference_cm = wheel_diameter_cm * 3.14159  # Omkredsen af hjulet (πd)

    # Antallet af rotationer der kræves for at køre den ønskede distance
    rotations_needed = abs(distance_cm / wheel_circumference_cm)

    # Reset motorerne for at starte fra nul
    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    correction_factor = 0.99  # Juster denne baseret på din robots kalibrering
    if (distance_cm>40):
        speed=400
    else:
        speed=200
    

    # Start arm motoren
    arm_motor.run(300)

    # Kør motorerne direkte uden brug af robot.drive
    while get_current_rotations() < abs(rotations_needed):
        #hvis ydermur
        if ultrasonic_sensor.distance() <180:
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_motor.stop()
            wait(100)
            robot.drive(50, 0)
            wait(750)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_collect(600,5000)
            print("Outer wall detected. Stopping the robot.")
            robot.drive(-1000, 0)
            wait(500)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            break
        #hvis kryds
        elif 188< ultrasonic_sensor.distance() < 194:
            print(ultrasonic_sensor.distance())
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_motor.stop()
            wait(300)
            robot.drive(50, 0)
            wait(500)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_collect(600,5000)
            print("cross  detected. Stopping the robot.")
            robot.drive(-1000, 0)
            wait(500)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            break

        # Juster hastigheden for at korrigere forskelle mellem motorerne
        left_speed = speed * correction_factor
        right_speed = speed

        left_motor.run(left_speed)
        right_motor.run(right_speed)

        wait(10)  # Kort pause for at undgå for hyppige opdateringer

    # Stop motorerne når måldistancen er nået
    robot.stop()
    left_motor.stop()
    right_motor.stop()
    arm_motor.stop()
    print("Final distance:", get_current_rotations() * wheel_circumference_cm)

def move_straight_corner(distance_cm):
    # Beregn antallet af rotationer, der kræves for at køre den ønskede distance
    wheel_diameter_cm = 4.32  # Diameteren på hjulet i cm
    wheel_circumference_cm = wheel_diameter_cm * 3.14159  # Omkredsen af hjulet (πd)

    # Antallet af rotationer der kræves for at køre den ønskede distance
    rotations_needed = abs(distance_cm / wheel_circumference_cm)

    # Reset motorerne for at starte fra nul
    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    correction_factor = 0.99  # Juster denne baseret på din robots kalibrering
    speed = 100 if distance_cm >= 0 else -100

    # Start arm motoren
    arm_motor.run(300)

    # Kør motorerne direkte uden brug af robot.drive
    while abs(get_current_rotations()) < abs(rotations_needed):
        if ultrasonic_sensor.distance() < 170:
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_motor.stop()
            wait(300)
            robot.drive(50, 0)
            wait(350)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            arm_collect(600,10000)
            print("Obstacle detected. Stopping the robot.")
            robot.drive(-1000, 0)
            wait(500)
            robot.stop()
            left_motor.stop()
            right_motor.stop()
            break

        # Juster hastigheden for at korrigere forskelle mellem motorerne
        left_speed = speed * correction_factor
        right_speed = speed

        left_motor.run(left_speed)
        right_motor.run(right_speed)

        wait(10)  # Kort pause for at undgå for hyppige opdateringer

    # Stop motorerne når måldistancen er nået
    robot.stop()
    left_motor.stop()
    right_motor.stop()
    arm_motor.stop()
    print("Final distance:", get_current_rotations() * wheel_circumference_cm)


# Call the calibration function at the start
calibrate_gyro(gyro_sensor)

def turning(degrees):
    arm_motor.run(300)

    #if degrees<0:
        #degrees=degrees*1.01
    robot.turn(degrees)
    arm_motor.stop()
    robot.stop()



# Handle command function
def handle_command(command):
    speed = 200
    time = 10000
    collectspeed = 600
    releasespeed = 600
    # Split the command into parts, because the first part is the actual command
    command_parts = command.split()
    command = command_parts[0]

    # Calibrate before critical commands
 

    if command == "COLLECT":
        arm_collect(collectspeed, time)

    if command == "ULTRA":
        while True:
            distance = ultrasonic_sensor.distance()
            print("Distance is: mm",(distance))
    elif command == "STOP":
        stop()
    elif command == "RELEASE":
        arm_release(releasespeed, time)
    elif command == "TURN2":
        if len(command_parts) > 1:
            degrees = command_parts[1]
            turn_by_degrees(float(degrees))
            response = "Command {} executed successfully.".format(command)
            conn.send(response.encode('utf-8'))
            return
    elif command == "TURN":
        if len(command_parts) > 1:
            degrees = command_parts[1]
            turning(float(degrees))
            response = "Command {} executed successfully.".format(command)
            conn.send(response.encode('utf-8'))
            return   
        
    elif command == "MOVE":
        if len(command_parts) > 1:
            distance = command_parts[1]
            rotations = move_straight(float(distance))
            response = "Command {} executed successfully. Rotations: {}".format(command, rotations)
            conn.send(response.encode('utf-8'))
            return
    elif command == "MOVE_COLLECT":
        if len(command_parts) > 1:
            distance = command_parts[1]
            rotations = move_straight_collect(float(distance))
            response = "Command {} executed successfully. Rotations: {}".format(command, rotations)
            conn.send(response.encode('utf-8'))
            return
        
    elif command == "MOVE_GOAL":
        if len(command_parts) > 1:
            distance = command_parts[1]
            rotations = move_straight_goal(float(distance))
            response = "Command {} executed successfully. Rotations: {}".format(command, rotations)
            conn.send(response.encode('utf-8'))
            return
    elif command == "MOVE_CORNER":
            if len(command_parts) > 1:
                distance = command_parts[1]
                rotations = move_straight_corner(float(distance))
                response = "Command {} executed successfully. Rotations: {}".format(command, rotations)
                conn.send(response.encode('utf-8'))
                return
    response = "Command {} executed successfully.".format(command)
    conn.send(response.encode('utf-8'))
    ev3.screen.print(response)

# Define the threshold distance.
threshold_distance = 30  # Adjust this value based on your requirements.

# Server settings
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 1024  # Port to listen on (non-privileged ports are > 1023)

# Write your program here.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
while True:
    print("Waiting for a new connection...")
    conn, addr = s.accept()
    # print(f"Connection established with {addr}")
    try:
        # Loop to handle commands from the current connection
        while True:
            data = conn.recv(1024)
            if not data:
                break  # Break the loop if no data is received, indicating the client has disconnected
            command = data.decode('utf-8')
            #     print(f"Received command: {command}")
            handle_command(command)
    finally:
        # Close the current connection before going back to listen for new ones
        #  print(f"Closing connection with {addr}")
        conn.close()


