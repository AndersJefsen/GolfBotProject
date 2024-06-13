import math

def get_corrected_position(before_robot_coordinates:list):
    x=before_robot_coordinates[0]
    y=before_robot_coordinates[1]
    map_width = 166.5
    map_height = 121.8
    robot_body_height = 30
    camera_x = 77  # Centered camera position (x-coordinate)
    camera_y = 51  # Centered camera position (y-coordinate)
    # Calculate distance from camera to robot's base (account for camera offset)
    distance_to_base = math.sqrt((x - camera_x)**2 + (y - camera_y)**2)

    # Ensure x and y are within map boundaries (assuming camera sees the whole map)
    if x < 0 or x > map_width:
        raise ValueError("X coordinate is outside map boundaries")
    if y < 0 or y > map_height:
        raise ValueError("Y coordinate is outside map boundaries")

    # Calculate angle between camera and top of robot
    angle = math.atan2(robot_body_height, distance_to_base)

    # Correct the x and y coordinates
    corrected_x = x + distance_to_base * math.cos(angle)
    corrected_y = y + distance_to_base * math.sin(angle)

    corrected_position_robot = [corrected_x, corrected_y]

    return corrected_position_robot


x_position = 77
y_position = 51

corrected_position = get_corrected_position(x_position, y_position)

print("Corrected position:", corrected_position)


x_position = 140
y_position = 100

corrected_position = get_corrected_position(x_position, y_position)

print("Corrected position:", corrected_position)



x_position = 20
y_position = 90

corrected_position = get_corrected_position(x_position, y_position)

print("Corrected position:", corrected_position)