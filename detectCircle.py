from ntpath import join
import cv2
from cv2 import getTrackbarPos
from cv2 import setIdentity
from matplotlib.pyplot import get
import numpy as np
import time
import pygame
from ballLoc import ballLoc
from pygame import mixer
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from curve import curve

#capture = cv2.VideoCapture("./pingpong.mp4")
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 768
height = 480

#RESIZES LIVE CAMERA FEED
capture.set(3, width)
capture.set(4, height)


if(not capture.isOpened()):
    print("ERROR")

#NEEDED FOR SLIDERS TO WORK
def nothing(a):
    pass

BALL_COLOR = "PINK"
DISPLAY_FPS = False

cv2.namedWindow('sliders', cv2.WINDOW_NORMAL)

if(BALL_COLOR == "ORANGE"):
    cv2.createTrackbar('param1', 'sliders', 44, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 14, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 1, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 37, nothing)
    cv2.createTrackbar('R1', 'sliders', 10, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 55, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 133, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 25, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 179, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 255, 255, nothing)
elif(BALL_COLOR == "WHITE"):
    cv2.createTrackbar('param1', 'sliders', 46, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 113, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 3, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 80, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 32, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 65, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 132, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 97, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 255, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 255, 255, nothing)
elif(BALL_COLOR == "GREEN"):
    cv2.createTrackbar('param1', 'sliders', 29, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 6, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 3, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 15, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 75, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 77, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 118, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 96, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 157, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 245, 255, nothing)
elif(BALL_COLOR == "PINK"):
    cv2.createTrackbar('param1', 'sliders', 27, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 8, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 2, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 35, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 125, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 84, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 130, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 171, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 238, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 248, 255, nothing)


cv2.createTrackbar('Table', 'sliders', 228, width, nothing)

cv2.createTrackbar('Net', 'sliders', 289, width, nothing)


prev_frame_time = 0
new_frame_time = 0

ballLocations1 = []
locCount1 = 0
bounceLocs = []

bounces_on_current_side = 0
has_moved_above_line = True

mixer.init()
beep = pygame.mixer.Sound('./beep.ogg')

def invert_y(y):
    return height - y

def quadratic(x, a, b, c):
    y = a * (x - b) * (x - b) + c
    return y

def getCurve(points):
    x = []
    y = []    
        
    for loc in points:
        x.append(loc.x)
        y.append(invert_y(loc.y))

    popt, pcov = curve_fit(quadratic, x, y)

    a = popt[0]
    b = popt[1]
    c = popt[2]

    table_intercept = quadratic(invert_y(tableLoc), a, b, c)

    return curve(a, b, c, table_intercept)


while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    # #IF USING PRE-RECORDED VIDEO#
    # percent = 50
    # width = int(img.shape[1] * percent / 100)
    # height = int(img.shape[0] * percent / 100)
    # dim = (width, height)

    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # height, width = img.shape[:2]
    # #############################
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerBound = np.array([cv2.getTrackbarPos('R1', 'sliders'),
    cv2.getTrackbarPos('G1', 'sliders'), cv2.getTrackbarPos('B1', 'sliders')])

    upperBound = np.array([cv2.getTrackbarPos('R2', 'sliders'),
    cv2.getTrackbarPos('G2', 'sliders'), cv2.getTrackbarPos('B2', 'sliders')])

    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    imgMasked = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('Masked image', imgMasked)

    gray = cv2.medianBlur(imgMasked, 5)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=1000, param1=cv2.getTrackbarPos('param1', 'sliders'), param2=cv2.getTrackbarPos('param2', 'sliders'), 
    minRadius=cv2.getTrackbarPos('min radius', 'sliders'), maxRadius=cv2.getTrackbarPos('max radius', 'sliders'))


    tableLoc = cv2.getTrackbarPos('Table', 'sliders')
    netLoc = cv2.getTrackbarPos('Net', 'sliders')

    cv2.line(img, (0, tableLoc), (width, tableLoc), (0, 255, 0), 3)
    cv2.line(img, (netLoc, 0), (netLoc, width), (0, 0, 255), 3)

    if circles is not None:
        for i in circles[0]:
            
            x = int(i[0])
            y = int(i[1])
            r = int(i[2])
            
            side = None

            if(x < netLoc):
                side = "Left"
            else:
                side = "Right"


            loc = ballLoc(x, y, r, time.time(), side, locCount1)
            ballLocations1.append(loc)
            locCount1 += 1

            if(len(ballLocations1) >= 2):
                if(ballLocations1[-2].side != ballLocations1[-1].side):
                    bounces_on_current_side = 0


            if(y >= (tableLoc - 20) and has_moved_above_line):
                if((bounces_on_current_side < 5)):
                    bounceLocs.append(loc)
                    beep.play()
                    has_moved_above_line = False
                    bounces_on_current_side += 1
            
            if(y < (tableLoc - 20)):
                has_moved_above_line = True
            

            #cv2.putText(img, f"X: {x} Y: {y} radius: {r}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            #draw Y Line
            cv2.line(img, (x, 0), (x, width), (255, 0, 0), 3)

            #draw X Line
            cv2.line(img, (0, y), (width, y), (255, 0, 0), 3)

            # draw the outer circle
            cv2.circle(img, (x, y), int(i[2]), (0, 0, 255), 2)
            
            # draw the center of the circle
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    


    if(DISPLAY_FPS):
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        fps = str(fps)
        cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    
    
    for loc in bounceLocs:
                cv2.circle(img, (loc.x, loc.y), 3, (0, 0, 255), 10)

    cv2.imshow('Ball Tracker', img)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        capture.release()
        cv2.destroyAllWindows()
        break


curves = []

if(len(bounceLocs) >= 1):

    pre_bounce_points = ballLocations1[0 : bounceLocs[0].index]
    curves.append(getCurve(pre_bounce_points))

    for i in range(len(bounceLocs) - 1):
        start_point  = bounceLocs[i].index
        end_point = bounceLocs[i + 1].index
        curves.append(getCurve(ballLocations1[start_point : end_point]))
    
    post_bounce_points = ballLocations1[bounceLocs[-1].index :]
    curves.append(getCurve(post_bounce_points))


for parab in curves:
    x_vals = np.arange(0, width, 1)
    y_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
    plt.plot(x_vals, y_vals, 'b')
    plt.ylim(ymin=invert_y(tableLoc), ymax=height)

x_points = []
y_points = []

for loc in ballLocations1:
    x_points.append(loc.x)
    y_points.append(invert_y(loc.y))

plt.plot(x_points, y_points, 'ok')
plt.vlines(x=netLoc, ymin=0, ymax=height, color='r', linestyle='-')
plt.show()


# pygame.init()
# screen = pygame.display.set_mode((width, height))
# screen.fill((0, 0, 255))
# running = True

# startTime = time.time()

# while running:
    
#     for x, y, r, t, s in ballLocations1:
#         netTime = time.time() - startTime
#         while(True):
#             pygame.draw.circle(screen, (0, 255, 0), (x, y), 3)
#             pygame.display.update()
#             break

        
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

