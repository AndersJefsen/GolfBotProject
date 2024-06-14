import math

def get_corrected_coordinates_robot(before_coordinates:list):
    x=before_coordinates[0] 
    y=before_coordinates[1]
    camera_height= 169
    robot_height = 30
    camera_x=77
    camera_y=51

    # Calculate distance from camera x,y to robot x,y (what the camera sees)
    distance = math.sqrt((x - camera_x) ** 2 + (y - camera_y) ** 2)
    print("distance",distance)
    #pythagoras to find the hypotenuse
    hypotenuse = math.sqrt(distance ** 2 + camera_height ** 2)

    #find smalle triangle
    h= camera_height-robot_height
    factor_triangle=h/camera_height
    print("factor_triangle",factor_triangle)

    l=distance*factor_triangle
    print("l",l)

    small_l = distance-l
    print("small_l",small_l)
    # Correcting the x-coordinate
    if x > camera_x:
        corrected_x = x - (small_l)  # Move right
    else:
        corrected_x = x + (small_l)  # Move left

    # Correcting the y-coordinate
    if y > camera_y:
        corrected_y = y - (small_l)  # Move up
    else:
        corrected_y = y + (small_l)  # Move down
        
    corrected_coordinates=[corrected_x,corrected_y]

    return corrected_coordinates

#examples
print("get_corrected_coordinates_robot([77.0605504587156, 62.9358024691358])")
print(get_corrected_coordinates_robot([77.0605504587156, 62.9358024691358]))
print("measured: 75.23 , 60.84")

print("get_corrected_coordinates_robot([149.2477, 110.4395])")
print(get_corrected_coordinates_robot([149.2477, 110.4395]))
print("measured: 136.1504, 98.7876")

print("get_corrected_coordinates_robot([148.029,10.3530])")
print(get_corrected_coordinates_robot([148.029,10.3530]))
print("measured: 133.4091, 16.3283")

print("get_corrected_coordinates_robot([28.526508226691043, 46.85820895522387])")
print(get_corrected_coordinates_robot([28.526508226691043, 46.85820895522387]))
print("measured: 37.4531,47.40")

print("get_corrected_coordinates_robot([3.9596, 113.0771])")
print(get_corrected_coordinates_robot([3.9596, 113.0771]))
print("measured: 14.9247, 103.1442")

print(get_corrected_coordinates_robot([120, 100]))
