import math

def get_correction_coordinate_robot(x, y, robot_height=31, camera_height=165, camera_x=81, camera_y=61):

    
    # Beregn den vandrette afstand B mellem punkt P og kameraet
    B = math.sqrt((camera_x - x)**2 + (camera_y - y)**2)
    
    if B == 0:
        return x, y
    # Beregn den vandrette afstand x fra punkt P til robotten
    x_robot = (B * robot_height) / camera_height
    
    # Beregn interpolationen for x og y koordinater
    R_x = x + (x_robot / B) * (camera_x - x)
    R_y = y + (x_robot / B) * (camera_y - y)
    
    return R_x, R_y



P_x = 81
P_y = 61
result = get_correction_coordinate_robot(P_x, P_y)
print(f"Punktet nedenfor robotten ved jorden har koordinaterne: ({result[0]:.2f}, {result[1]:.2f})")

P_x = 0
P_y = 0
result = get_correction_coordinate_robot(P_x, P_y)
print(f"Punktet nedenfor robotten ved jorden har koordinaterne: ({result[0]:.2f}, {result[1]:.2f})")


P_x = 161
P_y = 121
result = get_correction_coordinate_robot(P_x, P_y)
print(f"Punktet nedenfor robotten ved jorden har koordinaterne: ({result[0]:.2f}, {result[1]:.2f})")

P_x = 0
P_y = 121
result = get_correction_coordinate_robot(P_x, P_y)
print(f"Punktet nedenfor robotten ved jorden har koordinaterne: ({result[0]:.2f}, {result[1]:.2f})")

P_x = 161
P_y = 0
result = get_correction_coordinate_robot(P_x, P_y)
print(f"Punktet nedenfor robotten ved jorden har koordinaterne: ({result[0]:.2f}, {result[1]:.2f})")