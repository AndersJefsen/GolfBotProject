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
        self.cross = ArenaObject()
        self.wincap = None
        self.mode = None

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


class ArenaObject:
    def __init__(self):
        #con = contour, cord = coordinates, det = detections (number of detections)
        self.con = []
        self.cord = []
        self.det = 0
        self.recentlyDetected = False
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
