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

'''
convert to cartisian 

 @staticmethod
    def convert_orangeball_to_cartesian(image, orangeball_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()

        output_Image = image.copy()
        cartesian_coords = None

        if orangeball_contours:
            contour = orangeball_contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y))
                print(f"Orange Ball Cartesian Coordinates: {cartesian_coords}")
            # Tegner
            cv2.rectangle(output_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_Image, "1", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cartesian_coords, output_Image


         @staticmethod
    def convert_balls_to_cartesian(image, ball_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()

        cartesian_coords = []
        output_Image = image.copy()
        print(f"Found {len(ball_contours)} balls.")
        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords.append(ImageProcessor.convert_to_cartesian((center_x, center_y)))
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

            cv2.rectangle(output_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_Image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return cartesian_coords, output_Image


        @staticmethod
    def convert_robot_to_cartesian(output_image, robot_contours):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()
        robot_coordinates = []

        if robot_contours is None or len(robot_contours) < 3:
            print("Not enough blue dots found.")
            return robot_coordinates, output_image  
        
        for cnt in robot_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_image, (cX, cY), 3, (0, 255, 0), -1)  # Mark the blue dots on the image
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian((cX, cY))
                    robot_coordinates.append(cartesian_coords)
                    #print(f"Robot Cartesian Coordinates: {cartesian_coords}")


        return robot_coordinates, output_image
'''


'''
scale factor 
@staticmethod
    def calculate_scale_factors():
        bottom_left, bottom_right, top_left, top_right = ImageProcessor.corners.values()
        bottom_width = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        top_width = np.linalg.norm(np.array(top_left) - np.array(top_right))
        left_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        right_height = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        x_scale = 166 / max(bottom_width, top_width)
        y_scale = 121 / max(left_height, right_height)
        return x_scale, y_scale    
'''

'''
process image block


   #GKOPERGPOKEROPKG KOM NU JOHAN VS CODE STINKER
    @staticmethod
    def process_robot(indput_Image,output_Image):
        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = ImageProcessor.corners.values()
        midtpunkt = None
        angle = None

        contours = ImageProcessor.find_robot(indput_Image, min_size=0, max_size=100000)
       
        cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image,contours)
        
        if(contours is not None):
            midtpunkt,angle,output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
        
        
        return midtpunkt, angle, output_Image
    
        
@staticmethod
    def process_image(image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        edges = cv2.Canny(red, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = max(contours, key=cv2.contourArea)
        max_contour_area = cv2.contourArea(max_contour) * 0.99
        min_contour_area = cv2.contourArea(max_contour) * 0.002

        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]

        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 100)

        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
              ImageProcessor.corners['bottom_left'], ImageProcessor.corners['bottom_right'], \
        ImageProcessor.corners['top_left'], ImageProcessor.corners['top_right'] = \
            ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])

        # Calculate scale factors and update class attributes
        ImageProcessor.scale_factors['x_scale'], ImageProcessor.scale_factors['y_scale'] = \
            ImageProcessor.calculate_scale_factors()
        
        ImageProcessor.print_corner_info()

        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(image, [approx], 0, (60, 0, 0), 5)



        # robot_contour = ImageProcessor.find_robot(image, min_size=0, max_size=100000)
        # robot_coordinates = []

        # if robot_contour is not None:
        #     print("Found robot.")
        #     # Approximere konturen til en polygon og finde hjørnerne (spidserne)
        #     epsilon = 0.025 * cv2.arcLength(robot_contour, True)
        #     approx = cv2.approxPolyDP(robot_contour, epsilon, True)

        #     # Use k-means clustering to find the three most distinct points
        #     from sklearn.cluster import KMeans
        #     if len(approx) > 3:
        #         kmeans = KMeans(n_clusters=3)
        #         kmeans.fit(approx.reshape(-1, 2))
        #         points = kmeans.cluster_centers_.astype(int)
        #     else:
        #         points = approx

        #     for point in points:
        #         cv2.circle(image, tuple(point[0]), 5, (0, 255, 0), -1)
        #         if bottom_left_corner is not None:
        #             cartesian_coords = ImageProcessor.convert_to_cartesian(tuple(point[0]), bottom_left_corner,
        #                                                                    bottom_right_corner, top_left_corner,
        #                                                                    top_right_corner)
        #             robot_coordinates.append(cartesian_coords)
        #             print(f"Robot Kartesiske Koordinater: {cartesian_coords}")
        #     x, y, w, h = cv2.boundingRect(robot_contour)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        #     cv2.putText(image, "Robot", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # else:
        #     print("Ingen robot fundet.")

        #draw balls
        
        
        ball_contours = ImageProcessor.find_balls_hsv1(image, min_size=1000, max_size=2000)

        ImageProcessor.paintballs(ball_contours, "ball",image)

        roboball = ImageProcessor.find_robot(image, min_size=100, max_size=100000)
        ImageProcessor.paintballs(roboball, "robo ball", image)
 
        midpoint, direction = ImageProcessor.find_direction(roboball)
        if roboball and len(roboball) == 3:
            if midpoint and direction:
                    # Draw the direction from the midpoint
                endpoint = (midpoint[0] + direction[0], midpoint[1] + direction[1])
                cv2.circle(image, midpoint, 10, (0, 0, 255), -1)  # Red dot at midpoint
                cv2.line(image, midpoint, endpoint, (255, 0, 0), 3)  # Blue line indicating direction
                midpoint_cm=ImageProcessor.convert_to_cartesian(midpoint)
                print("Midpoint:", midpoint_cm)
                print("Direction to third point:", direction)
                # cv2.imshow('Directional Image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        else:
            print("Could not find exactly three balls., found ", len(roboball))       

        orange_ballcontours = ImageProcessor.find_orangeball_hsv(image, min_size=1000, max_size=2000)
        if orange_ballcontours:
            contour = orange_ballcontours[0]
            print(contour)
            ImageProcessor.paintballs(contour,"orangeball", image)


        big_ball_contour = ImageProcessor.find_bigball_hsv(image, min_size=4000, max_size=10000)
        if big_ball_contour is not None:
            ImageProcessor.paintballs(orange_ballcontours, "egg", image) 

        angle = ImageProcessor.calculate_angle(direction)
        
        if angle is not None:
            print(f"Angle: {angle} degrees")
        cross_contours = ImageProcessor.find_cross_contours(filtered_contours,image)
        for i, cnt in enumerate(cross_contours):
            cv2.drawContours(image, [cnt], 0, (255, 0, 0), 3)
            for point in cnt:
                x, y = point.ravel()
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1])) for point in cnt]

                #print(f"Cross {i+1} Cartesian Coordinates: {cartesian_coords}")

        print(f"Found {len(cross_contours)} crosses.")

        # Mark the corners on the output_image for the arena
        if bottom_left_corner is not None:
            cv2.circle(image, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(image, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(image, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(image, top_right_corner, 10, (255, 0, 135), -1)

        # cv2.imshow('image2', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        @staticmethod
        def process_robot(indput_Image, output_Image):
            midtpunkt = None
            angle = None
            contours = ImageProcessor.find_robot(indput_Image)
            cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image, contours,)
            if (contours is not None):
                midtpunkt, angle, output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
            return midtpunkt, angle, output_Image

    @staticmethod
    def process_robotForTesting(indput_Image, output_Image):
        midtpunkt = None
        angle = None
        contours = ImageProcessor.find_robot(indput_Image)
        cartesian_coords, output_Image = ImageProcessor.convert_robot_to_cartesian(output_Image, contours)
        ImageProcessor.paintballs(contours, "robot", image)
        if (contours is not None):
            midtpunkt, angle, output_Image = ImageProcessor.calculate_robot_midpoint_and_angle(contours, output_Image)
        return midtpunkt, angle, output_Image, contours
    
    
   
    ### TEST FUNKTION FOR AT SE OM DET VIRKER
    def process_image(image):
        success,outputimage, bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner, filtered_contours = ImageProcessor.find_Arena(
            image, image.copy())
        if not success:
            print("Could not find the arena.")
            return
        ImageProcessor.showimage('arena', outputimage)
        
        cross_counters, output_image_with_cross = ImageProcessor.find_cross_contours( filtered_contours, outputimage)
        cartesian_cross_list, output_image_with_cross = ImageProcessor.convert_cross_to_cartesian(cross_counters, outputimage)
        # Create the mask using the detected arena corners
        ImageProcessor.showimage('cross', outputimage)

        arenaCorners = [bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner]

        balls_contour = ImageProcessor.find_balls_hsv1(outputimage, 1000,2000)
        ball_list, outputimage = ImageProcessor.convert_balls_to_cartesian(outputimage,balls_contour)
        outputimage = ImageProcessor.paintballs(balls_contour, "ball", outputimage)
        ImageProcessor.showimage('balls', outputimage)
   

        #midtpunkt, angle, output_image_with_robot, contours = ImageProcessor.process_robotForTesting(output_image_with_balls,
        #                                                                                            output_image_with_balls.copy())
        ImageProcessor.process_robotForTesting(outputimage, outputimage)
        ImageProcessor.showimage('robot', outputimage)
        
        orangeball_contour = ImageProcessor.find_orangeball_hsv(outputimage, 1000,2000)
        if(orangeball_contour):
            outputimage=ImageProcessor.paintballs(orangeball_contour, "orange", outputimage)
            ImageProcessor.showimage('orange', outputimage)


        egg_contour = ImageProcessor.find_bigball_hsv(outputimage, 5000,10000,0.3,1.8)
        if(egg_contour):
            outputimage=ImageProcessor.paintballs(egg_contour, "egg", outputimage)
            ImageProcessor.showimage('final', outputimage)

            

            @staticmethod
    def createMask(imageToDetectOn, points):
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(imageToDetectOn.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return mask

    @staticmethod
    def useMask(imageToMask, mask):
        if mask.shape[:2] != imageToMask.shape[:2]:
            raise ValueError("The mask and the image must have the same dimensions")
        return cv2.bitwise_and(imageToMask, imageToMask, mask=mask)

'''