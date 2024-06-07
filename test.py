from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port

# Initialize the EV3 Brick.
ev3 = EV3Brick()

# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Run the motors at 500 degrees per second for 2 seconds.
left_motor.run_time(500, 2000)
right_motor.run_time(500, 2000)