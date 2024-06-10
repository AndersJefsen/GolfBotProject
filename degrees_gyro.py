
#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, TouchSensor, ColorSensor, InfraredSensor, UltrasonicSensor, GyroSensor
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile



# Initialize the EV3 Brick.
ev3 = EV3Brick()

# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Initialize the drive base.
robot = DriveBase(left_motor, right_motor, wheel_diameter=4.32, axle_track=13.3)

# Set the speed and acceleration.




# Initialize the gyro sensor.
gyro_sensor = GyroSensor(Port.S1)

# Reset the gyro sensor to zero.
gyro_sensor.reset_angle(0)

# Calculate the target angle.
target_angle = -90


# Turn until the gyro sensor's angle is equal to the target angle.
def turn_to_target(robot, gyro_sensor, target_angle):
    # Reset the gyro sensor to robot angle
    gyro_sensor.reset_angle(0)

    # Turn until the gyro sensor's angle is equal to the target angle.
    while abs(gyro_sensor.angle()) < abs(target_angle):
        # Turn the robot based on the angle difference.
        robot.drive(0, 40)

    # Stop the robot.
    robot.stop()

# Example usage:
turn_to_target(robot, gyro_sensor, target_angle)
