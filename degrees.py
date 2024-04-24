
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
robot.settings(turn_rate=40, turn_acceleration=40)



#correction_factor_turning_turning_clockwise = 1.04
correction_factor_turning_anticlockwise = 1.045
degrees=-720

# Turn clockwise by 360 degrees and back again.
robot.turn((correction_factor_turning_anticlockwise*degrees))
