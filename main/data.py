import numpy as np
import time
import cv2 as cv
class Data:
   

    def __init__(self):
        #[0] bottom left, [1] bottom right, [2] top right, [3] top left
        self.arenaCorners = []
        self.whiteballs = []
        self.orangeBall = ArenaObject()
        self.robotPositions = []
        self.egg = ArenaObject()
        self.arenaMask = None
        self.socket = None
        self.robot = Robot()
        self.cross = cross()
        self.helpPoints = []
        self.outerArea = OuterArea()
        self.drivepoints = []
        self.mode = None
        self.wincap = None
        self.testpicturename = None
        self.screenshot = None
        self.output_image = None



    def addBalls(self, contours, cordinates):
        for contour, cord in zip(contours, cordinates):
            ballexists = False
            for ball in self.whiteballs:
                # Check if the ball coordinates are within a small distance (epsilon) from each other
                epsilon = 1.5  # You might need to adjust this threshold
                if abs(cord[0] - ball.cord[0]) <= epsilon and abs(cord[1] - ball.cord[1]) <= epsilon:
                    if ball.det < 15:
                        ball.det += 1
                    
                    ballexists = True
                    ball.recentlyDetected = True
                    break
            
            if not ballexists:
                newball = ArenaObject()
                newball.con = contour
                newball.cord = cord
                newball.det = 1  # Initialize detection count
                newball.recentlyDetected = True
                self.whiteballs.append(newball)

        # Remove balls that were not detected recently
        for ball in self.whiteballs:
            if not ball.recentlyDetected:
                if ball.det > 0:
                    ball.det -= 1
                else:
                    self.whiteballs.remove(ball)
            else:
                ball.recentlyDetected = False
  
    def printBalldetections(self):
        for index, ball in enumerate(self.balls):
            print(f"Ball index: {index}, Coordinates: {ball.cord}, Number of detections: {ball.det}")
    def printBalls(self):
        for index, ball in enumerate(self.whiteballs):
            print(f"Ball index: {index}, Coordinates: {ball.cord}, Number of detections: {ball.det}")

    def getBallPositions(self):
        return self.ballPositions
    def addArenaCorners(self, corners):
        self.arenaCorners = corners
    def getArenaCorners(self):
        return self.arenaCorners
    def addArenaMask(self, mask):
        self.arenaMask = mask
    def getAllBallCordinates(self):
        return [ball.cord for ball in self.whiteballs]
    def getAllBallContours(self):
        return [ball.con for ball in self.whiteballs]
    def resetRobot(self):
        self.robot = Robot()
    def getAllHelpPointsCon(self):
        return [hp.con for hp in self.helpPoints]
    
    def add_cross_factor(self,factor,corner,center):
        direction_vector = np.array(corner) - np.array(center)
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            raise ValueError("Center and corner cannot be the same point")
        normalized_vector = direction_vector / norm
        addition = factor * normalized_vector
        new_point = np.array(corner) + addition
    
    # Extract the x and y additions for output if needed
        
    
        return new_point
        
    def areal(self,x1, y1, x2, y2, x3, y3):
 
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) 
                    + x3 * (y1 - y2)) / 2.0)
    
    def is_point_in_triangle_cross_product(self,p, a, b, c):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        x, y = p
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        A = self.areal (x1, y1, x2, y2, x3, y3)

        # Calculate area of triangle PBC 
        A1 = self.areal (x, y, x2, y2, x3, y3)
        
        # Calculate area of triangle PAC 
        A2 = self.areal (x1, y1, x, y, x3, y3)
        
        # Calculate area of triangle PAB 
        A3 = self.areal (x1, y1, x2, y2, x, y)
        
        # Check if sum of A1, A2 and A3 
        # is same as A
        if(A == A1 + A2 + A3):
            return True
        else:
            return False

    def find_HP(self):
        self.outerArea = OuterArea()
        print("dbug1")
        if self.cross.corner_con is not None:
            print("debug12")
            self.find_Cross_HP()
        print("dbug2")
        self.find_Corner_HP()
        print("dbug3")
        self.outerArea.create_areas(self.arenaCorners[0],self.arenaCorners[1],self.arenaCorners[2],self.arenaCorners[3])
        for whiteball in self.whiteballs:
            x, y, w, h = cv.boundingRect(whiteball.con)
            center_x = x + w // 2
            center_y = y + h // 2
            in_outer_area = False
            factor = 150
            for area in self.outerArea.areas:
                if area.type == "cross_corner":
                    if(self.is_point_in_triangle_cross_product((center_x,center_y),area.points[0],area.points[1],area.points[2])):
                        in_outer_area = True

                        new_point = self.add_cross_factor(factor,area.points[1],self.cross.center)
                        print("newpoint: ",new_point)
                        print("x: ",center_x," y: ",center_y)
                        helpPointCord = HelpPoint((new_point[0],new_point[1]),whiteball)
                       
                        
                       
                        self.helpPoints.append(helpPointCord)
                        
                else:      
                    max_x = max([p[0] for p in area.points])
                    min_x = min([p[0] for p in area.points])
                    max_y = max([p[1] for p in area.points])
                    min_y = min([p[1] for p in area.points])
                
                    if center_x > min_x and center_x < max_x and center_y > min_y and center_y < max_y:
                        in_outer_area = True
                        x_addision = 0
                        y_addision = 0
                        
                        if area.type == "BL_corner":
                            x_addision =  factor
                            y_addision = -factor
                        elif area.type == "BR_corner":
                            x_addision = - factor
                            y_addision = - factor
                        elif area.type == "TR_corner":
                            x_addision = - factor
                            y_addision =  factor
                        elif area.type == "TL_corner":
                            x_addision =  factor
                            y_addision =  factor
                        elif area.type == "left_side":
                            x_addision =  factor
                        elif area.type == "right_side":
                            x_addision = - factor
                        elif area.type == "top_side":
                            y_addision =  factor
                        elif area.type == "bottom_side":
                            y_addision = - factor
                    
                    
                        whiteball_center = np.array([center_x, center_y])
                        helpPointCord = HelpPoint((whiteball_center[0] + x_addision, whiteball_center[1] + y_addision),whiteball)
                        self.helpPoints.append(helpPointCord)
                    
                        
                        break
            if in_outer_area == False:
                whiteball_center = np.array([center_x, center_y])
                helpPointCord = HelpPoint((whiteball_center[0], whiteball_center[1]),whiteball)
                self.helpPoints.append(helpPointCord)

    
    def find_Corner_HP(self):
        if not self.arenaCorners:
            return False
    
        
        num_corners = len(self.arenaCorners)
        self.drivepoints = []
        for i in range(num_corners):
            prev_corner = self.arenaCorners[i - 1]
            current_corner = self.arenaCorners[i]
            next_corner = self.arenaCorners[(i + 1) % num_corners]

            # Calculate vectors
            vec1 = np.array(prev_corner) - np.array(current_corner)
            vec2 = np.array(next_corner) - np.array(current_corner)

            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)

            # Calculate the bisector vector
            bisector = vec1_norm + vec2_norm
            bisector_norm = bisector / np.linalg.norm(bisector)

            # Define the length of the help point line (extend into the arena)
            length = 300  # You can adjust this length as needed

            # Calculate the help point position
            help_point = np.array(current_corner) + bisector_norm * length
            self.drivepoints.append(help_point.tolist())

    def extend_vector(self,vec, len_x, len_y):
            return np.array([
                vec[0] * len_x / np.abs(vec[0]),
                vec[1] * len_y / np.abs(vec[1])
            ])    
    def find_Cross_HP(self):

       
        
        if  self.cross.corner_con is None:
            print("Error: contour_points is empty.")
            return False


        self.cross.calculate_center()
        center = np.array(self.cross.center)

        # Calculate distances from the center to all points
        distances = []
        for i, point in enumerate(self.cross.corner_con):
            distance = np.linalg.norm(center - np.array(point[0]))
            distances.append((distance, point[0], i))
            
        # Sort points by distance and select the four closest
        distances.sort()
       

       
        for _, corner, idx in distances[:4]:
            prev_idx = (idx - 1) % len(self.cross.corner_con)
            next_idx = (idx + 1) % len(self.cross.corner_con)
            
            area_points = [
                self.cross.corner_con[prev_idx][0],
                corner,
                self.cross.corner_con[next_idx][0]
            ]

            self.outerArea.areas.append(Area(area_points, "cross_corner"))

    
      
      


    

class ArenaObject:
    def __init__(self):
        #con = contour, cord = coordinates, det = detections (number of detections)
        self.con = []
        self.cord = []
        self.det = 0
        self.recentlyDetected = False
        self.angle = None



class cross(ArenaObject):
    def __init__(self):
        super().__init__()
        self.corner_con = []
        self.center = None
    def calculate_center(self):
        # Calculate the center of the cross
        if self.corner_con is None:
            print("Error: contour_points is empty.")
            return (0, 0)

        #print("Calculating center")
        try:
            x_coords = [p[0][0] for p in self.corner_con]
            y_coords = [p[0][1] for p in self.corner_con]
        except IndexError as e:
            print(f"Error accessing point coordinates: {e}")
            return (0, 0)

        center_x = sum(x_coords) / len(self.corner_con)
        center_y = sum(y_coords) / len(self.corner_con)

        #print(f"Center calculated at: ({center_x}, {center_y})")
        center = (center_x, center_y)
        self.center = center
    
class Robot:
    def __init__(self):
        #con = contour, cord = coordinates, det = detections (number of detections)
        self.balls = []
        self.ballcontours = []
        self.originalMidtpoint = None
        self.compare = []
        self.midpoint = None
        self.direction = None
        self.angle = None
        self.det = 0
        self.detected = False
        self.min_detections = 1

    def add_detection(self, midpoint, angle):
        # Add new detection to the compare list
        exists = False
        epsilon = 5  # Adjust epsilon as needed
        for i, (mp, ang, det) in enumerate(self.compare):
            if abs(midpoint[0] - mp[0]) <= epsilon and abs(midpoint[1] - mp[1]) <= epsilon:
                self.compare[i] = (mp, ang, det + 1)
                exists = True
                break
        if not exists:
            self.compare.append((midpoint, angle, 1))

    def get_best_robot_position(self):
        # Find the detection with the highest count
        best_position = None
        highest_detection = self.min_detections

        for midpoint, angle, det in self.compare:
            if det >= highest_detection:
                best_position = (midpoint, angle)
                highest_detection = det

        return best_position

    def set_min_detections(self, min_detections):
        self.min_detections = min_detections



class HelpPoint:
    def __init__(self,con,ball = None):
        self.con = con
        self.ball = ball

class OuterArea:
    def __init__(self):
        self.areas = []
    def create_areas(self,bottomleft,bottomright,topright,topleft):
        cornerLength = 100
        # Calculate the correct points based on bottomleft and cornerLength
        

        # Create an Area object with these four calculated corners
        bot_left_corner_points = self.create_corners(bottomleft, "bottom-left", cornerLength)    
        self.areas.append(Area(bot_left_corner_points, "BL_corner"))
        bot_right_corner_points = self.create_corners(bottomright, "bottom-right", cornerLength)
        self.areas.append(Area(bot_right_corner_points, "BR_corner"))
        top_right_corner_points = self.create_corners(topright, "top-right", cornerLength)
        self.areas.append(Area(top_right_corner_points, "TR_corner"))
        top_left_corner_points = self.create_corners(topleft, "top-left", cornerLength)
        self.areas.append(Area(top_left_corner_points, "TL_corner"))

        left_side = self.create_sides(bottomleft, topleft, "left", cornerLength)
        self.areas.append(Area(left_side, "left_side"))
        right_side = self.create_sides(bottomright, topright, "right", cornerLength)
        self.areas.append(Area(right_side, "right_side"))
        top_side = self.create_sides(topleft, topright, "top", cornerLength)
        self.areas.append(Area(top_side, "top_side"))
        bottom_side = self.create_sides(bottomleft, bottomright, "bottom", cornerLength)
        self.areas.append(Area(bottom_side, "bottom_side"))
    
    def create_sides(self,cornerPointOne,cornerPointTwo,side,conerLength):
        if side == "left":
            # Calculate the correct points based on bottomleft and cornerLength
            bottom_left = (cornerPointOne[0], cornerPointOne[1]-conerLength)
            bottom_right = (cornerPointOne[0]+conerLength, cornerPointOne[1]-conerLength)
            top_right = (cornerPointTwo[0]+conerLength, cornerPointTwo[1] + conerLength)
            top_left = (cornerPointTwo[0], cornerPointTwo[1] + conerLength)
        elif side == "right":
            # Calculate the correct points based on bottomright and cornerLength
            bottom_left = (cornerPointOne[0]-conerLength, cornerPointOne[1]-conerLength)
            bottom_right = (cornerPointOne[0], cornerPointOne[1]-conerLength)
            top_right = (cornerPointTwo[0], cornerPointTwo[1] + conerLength)
            top_left = (cornerPointTwo[0]-conerLength, cornerPointTwo[1] + conerLength)
        elif side == "top":
            # Calculate the correct points based on topright and cornerLength
            bottom_left = (cornerPointOne[0]+conerLength, cornerPointOne[1]+conerLength)
            bottom_right = (cornerPointTwo[0]-conerLength, cornerPointTwo[1]+conerLength)
            top_right = (cornerPointTwo[0]-conerLength, cornerPointTwo[1])
            top_left = (cornerPointOne[0]+conerLength, cornerPointOne[1])
        elif side == "bottom":
            # Calculate the correct points based on topleft and cornerLength
            bottom_left = (cornerPointOne[0]+conerLength, cornerPointOne[1])
            bottom_right = (cornerPointTwo[0]-conerLength, cornerPointTwo[1])
            top_right = (cornerPointTwo[0]-conerLength, cornerPointTwo[1]-conerLength)
            top_left = (cornerPointOne[0]+conerLength, cornerPointOne[1]-conerLength)
        else:
            raise ValueError("The corner must be either bottom-left, bottom-right, top-right, or top-left")
        return bottom_left,bottom_right,top_right,top_left


    def create_corners(self,cornerPoint,corner,conerLength):
        cornerLength = 100
        if corner == "bottom-left":
            # Calculate the correct points based on bottomleft and cornerLength
           bottom_left = cornerPoint
           bottom_right = (cornerPoint[0] + cornerLength, cornerPoint[1])
           top_right = (cornerPoint[0] + cornerLength, cornerPoint[1] - cornerLength)
           top_left = (cornerPoint[0], cornerPoint[1] - cornerLength)
        elif corner == "bottom-right":
            # Calculate the correct points based on bottomright and cornerLength
            bottom_left = (cornerPoint[0]-cornerLength, cornerPoint[1])
            bottom_right = cornerPoint
            top_right = (cornerPoint[0], cornerPoint[1] - cornerLength)
            top_left = (cornerPoint[0]-cornerLength, cornerPoint[1] - cornerLength)
        elif corner == "top-right":
            # Calculate the correct points based on topright and cornerLength
            bottom_left = (cornerPoint[0]-cornerLength, cornerPoint[1] + cornerLength)
            bottom_right = (cornerPoint[0], cornerPoint[1] + cornerLength)
            top_right = cornerPoint
            top_left = (cornerPoint[0] - cornerLength, cornerPoint[1])
        elif corner == "top-left":
            # Calculate the correct points based on topleft and cornerLength
            bottom_left = (cornerPoint[0], cornerPoint[1] + cornerLength)
            bottom_right = (cornerPoint[0] + cornerLength, cornerPoint[1] + cornerLength)
            top_right = (cornerPoint[0] + cornerLength, cornerPoint[1])
            top_left = cornerPoint
        else:
            raise ValueError("The corner must be either bottom-left, bottom-right, top-right, or top-left")
        return bottom_left,bottom_right,top_right,top_left
                
class Area:
    def __init__(self, points, type="area"):
        if len(points) == 4 or len(points) == 3:
           self.points = points
           self.type = type 
        else:
            raise ValueError("The area must have four points")
    def get_points(self):
        return self.points