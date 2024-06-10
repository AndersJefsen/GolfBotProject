    angle_radians = np.arctan2(direction_vector[1], direction_vector[0])
            
            # Convert angle to degrees
            angle_degrees = np.degrees(angle_radians)
            
            # Normalize angle to be between 0 and 360
            angle_degrees = angle_degrees % 360
            
            return angle_degrees