from path import find_closest_ball
from degrees_gyro import turn_to_target

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, TouchSensor, ColorSensor, InfraredSensor, UltrasonicSensor, GyroSensor
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile



robot_position = (0, 0)  # Example robot position get pos
robot_angle=0 #get angle
balls = [(3, 4), (1, 2), (5, 5)]  # Example balls positions get balls
closest_ball, distance_to_ball, angle_to_turn = find_closest_ball(robot_position, balls, robot_angle)

print(closest_ball, distance_to_ball, angle_to_turn)


ev3 = EV3Brick()

# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Initialize the drive base.
robot = DriveBase(left_motor, right_motor, wheel_diameter=4.32, axle_track=13.3)
gyro_sensor = GyroSensor(Port.S1)


gyro_sensor.reset_angle(0)


turn_to_target(robot, gyro_sensor, angle_to_turn)

