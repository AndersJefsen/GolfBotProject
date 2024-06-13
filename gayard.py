#computer vision

"""
    #find egg
    @staticmethod
    def find_bigball_threshold(image, min_size=300, max_size=1000000000, threshold=180):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the grayscale image
        _, thresh_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

        # Display the processed image
        cv2.imshow('Processed Image', thresh_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_ball_contour = None

        # Logic to find the biggest ball
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.0001 <= circularity <= 500:
                    big_ball_contour = cnt
                    max_area = area

        return big_ball_contour

#find white balls
    @staticmethod
    def find_balls_threshold(image, min_size=300, max_size=1000000000, threshold=180):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the grayscale image
        _, thresh_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

        # Display the processed image
        cv2.imshow('Processed Image', thresh_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []

        # Logic to find contours of the balls
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    ball_contours.append(cnt)

        return ball_contours

 
    @staticmethod
    def find_robot_withOutput(inputImage,outPutImage, min_size=100, max_size=100000,bottom_left_corner=None, bottom_right_corner=None, top_left_corner=None, top_right_corner=None):
        hsv_image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([78,100,100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")

        # Threshhold the HSV image image to get only blue colors
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
       
        # Load maskerne på billedet

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet

        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []
        # Logikken for at finde countours på boldene
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    ball_contours.append(cnt)

        # Converter til cartesian coordinates
        roboball = ball_contours 
        
        for i, contour in enumerate(roboball, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(outPutImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(outPutImage, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                #print(f"roboBall {i} Cartesian Coordinates: {cartesian_coords}")
                
        
        midpoint, direction = ImageProcessor.find_direction(roboball)
        if roboball and len(roboball) == 3:
            if midpoint and direction:
                    # Draw the direction from the midpoint
                endpoint = (midpoint[0] + direction[0], midpoint[1] + direction[1])
                cv2.circle(outPutImage, midpoint, 10, (0, 0, 255), -1)  # Red dot at midpoint
                cv2.line(outPutImage, midpoint, endpoint, (255, 0, 0), 3)  # Blue line indicating direction
                #print("Midpoint:", midpoint)
                #print("Direction to third point:", direction)
                
        #else:
            #print("Could not find exactly three balls., found ", len(roboball))        
        

        angle = None
        if direction:
            angle = ImageProcessor.calculate_angle(direction)
        
        if angle is not None:
            print(f"Angle: {angle} degrees")
        print("here")
        return ball_contours, outPutImage, angle, midpoint
        
        """