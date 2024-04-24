import math

balls=[(1,1), (2,2), (3,3)] # get from cv
goal=(4,4) #get from cv
robot=(0,0) #get from cv

def calcdist(point1, point2):
    #pythagoras
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def nextball(robot, balls):
#next neighbour not avoiding obsticle
    nearest = None
    mindist = float('inf')
    for ball in balls:
        dist = calcdist(robot, ball)
        if dist < mindist:
            mindist = dist
            nearest = ball
    return nearest

def align(robot, target_position):
   #zohan
    pass

def move(robot, target_position):
   #zohan
    pass

def pickup():
    #zohan
    pass
def drop():
    #zohan
    pass

while balls:
    targetball = nextball(robot, balls)
    distance = calcdist(robot, targetball)
    if distance < 0.5:  
                pickup()
                #maybe balls.remove(nextball) 
    else:
        align(robot, targetball)
        move(robot, targetball)

move(robot, goal)
drop()