
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, GyroSensor
from pybricks.parameters import Port
from pybricks.robotics import DriveBase

# Initialize the EV3 Brick, motors, and gyro sensor
ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
gyro_sensor = GyroSensor(Port.S1)

# Set the drive base with the motors and wheel diameter
wheel_diameter = 4.32
axle_track = 13.3
robot = DriveBase(left_motor, right_motor, wheel_diameter, axle_track)
    # Calibrate the gyro sensor
gyro_sensor.reset_angle(0)

desired_angle=0

speed=10

# Define the correction factor for distance.

# Drive straight using the gyro sensor
while True:
    # Calculate the angle difference
    angle_difference = desired_angle - gyro_sensor.angle()

    # Correct the steering based on the angle difference
    robot.drive(speed, angle_difference * 1.2)

    # Wait for a short time

    # Calculate the distance traveled
    distance_traveled = robot.distance() 

    # Check if the robot has traveled 100cm
    if distance_traveled >= 206:
        # Stop the robot
        robot.stop()
        break