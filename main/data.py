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

    def addBalls(self, contours, cordinates):
        #print("Adding balls")
        for contour, cord in zip(contours, cordinates):
            ballexists = False
            for ball in self.whiteballs:
                if (cord[0] < ball.cord[0]+5 or cord[0]> ball.cord[0]-5) and (cord[1] > ball.cord[1]-5 or cord[1] < ball.cord[1]+5):
                    if ball.det < 30:
                        ball.det += 1
                    ballexists = True
                    ball.recentlyDetected = True
                    break
            if not ballexists:
                newball = ArenaObject()
                newball.con = contour
                newball.cord = cord
                newball.det += 1
                newball.recentlyDetected = True
                self.whiteballs.append(newball)
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
        self.midpoint = None
        self.direction = None
        self.angle = None
        self.det = 0
        self.detected = False