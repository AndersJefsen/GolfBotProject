class Data:
   

    def __init__(self):
        #[0] bottom left, [1] bottom right, [2] top right, [3] top left
        self.arenaCorners = []
        self.balls = []
        self.robotPositions = []
        self.arenaMask = None
        self.socket = None

    def addBalls(self, contours, cordinates):
        print("Adding balls")
        for contour, cord in zip(contours, cordinates):
            ballexists = False
            for ball in self.balls:
                if (cord[0] < ball.cord[0]+5 or cord[0]> ball.cord[0]-5) and (cord[1] > ball.cord[1]-5 or cord[1] < ball.cord[1]+5):
                    if ball.det < 30:
                        ball.det += 1
                    ballexists = True
                    ball.recentlyDetected = True
                    break
            if not ballexists:
                newball = ArenaObject(contour, cord)
                newball.det += 1
                newball.recentlyDetected = True
                self.balls.append(newball)
        for ball in self.balls:
            if not ball.recentlyDetected:
                if ball.det > 0:
                    ball.det -= 1
                else:
                    self.balls.remove(ball)
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

class ArenaObject:
    def __init__(self, contours, cordinates,detections =0):
        self.con = contours
        self.cord = cordinates
        self.det = detections
        self.recentlyDetected = False