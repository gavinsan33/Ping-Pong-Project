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
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import * 
import math
import sys
from turn import turn

#capture = cv2.VideoCapture("./pingpong.mp4")
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 768
height = 480

#RESIZES LIVE CAMERA FEED
capture.set(3, width)
capture.set(4, height)


assert capture.isOpened()

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
    cv2.createTrackbar('param1', 'sliders', 27, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 8, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 3, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 80, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 130, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 85, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 228, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 255, 255, nothing)
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
    cv2.createTrackbar('param2', 'sliders', 3, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 1, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 40, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 125, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 106, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 130, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 171, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 238, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 248, 255, nothing)

cv2.createTrackbar('Table', 'sliders', 194, width, nothing)

cv2.createTrackbar('Net', 'sliders', 289, width, nothing)

cv2.createTrackbar('Ignore', 'sliders', 230, width, nothing )

prev_frame_time = 0
new_frame_time = 0

ballLocations1 = []
locCount1 = 0
bounceLocs = []
turns = []
bounces_on_current_side = 0
has_moved_above_line = True

left_score = 0
right_score = 0

first_of_turn = True

mixer.init()
beep = pygame.mixer.Sound('./beep.ogg')
boing = pygame.mixer.Sound("./boing.ogg")

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

    ###TAKEN DIRECTLY FROM STACK OVERFLOW###
    #CALCULATES CORRELATION COEFFICIENT
    residuals = y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    #######################################

    #print(r_squared)

    # if(r_squared < 0.5):
    #     return None

    a = popt[0]
    b = popt[1]
    c = popt[2]

    table_intercept = x[-1]

    delta_time = points[-1].time - points[0].time

    return curve(a, b, c, x[0], table_intercept, delta_time)

def Cube(len, width, height, pos, color, outline):

    x, y, z = pos
    
    verticies = (

    (len + x, -width + y, -height + z),
    (len + x, width + y, -height + z),
    (-len + x, width + y, -height + z),
    (-len + x, -width + y, -height + z),
    (len + x, -width + y, height + z),
    (len + x, width + y, height + z),
    (-len + x, -width + y, height + z),
    (-len + x, width + y, height + z),
    
    )

    edges = (

        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 7),
        (6, 3),
        (6, 4),
        (6, 7),
        (5, 1),
        (5, 4),
        (5, 7),
    )
    
    surfaces = (

        (0, 1, 2, 3),
        (3, 2, 7, 6),
        (6, 7, 5, 4),
        (4, 5, 1, 0),
        (1, 5, 7, 2),
        (4, 0, 3, 6),

    )
    
    glBegin(GL_QUADS)

    for s in surfaces:
        glColor3fv(color)
        for vertex in s:
            glVertex3fv(verticies[vertex])


    glEnd()

    glBegin(GL_LINES)

    for edge in edges:
        for vertex in edge:
            if(outline): glColor3fv((1, 1, 1))
            glVertex3fv(verticies[vertex])

    glEnd()

def Circle(radius, resolution, pos, color, outline):
    x, y, z = pos
    
    vert_list = []
    DEGREES_PER_VERT = 360.0 / resolution

    vert_list.append((x, y, z))

    for i in range(resolution):
        degrees = i * DEGREES_PER_VERT
        cos = math.cos(degrees)
        sin = math.sin(degrees)

        vert_list.append((radius * cos + x, radius * sin + y, z))

    verticies = tuple(vert_list)

    surfaces = []

    for i in range(1, resolution):
        surfaces.append((0, i, i + 1))


    glBegin(GL_TRIANGLES)
    glColor3fv(color)

    for s in surfaces[1:]:
        for vertex in s:
            glVertex3fv(verticies[vertex])
    
    glEnd()

def update_3D_render(ballX, ballY, ballZ):
    #glRotatef(0.25, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    #FLOOR
    #Circle(15, 500, (0, 0, 0), (0, 1, 0.25), False)

    #TABLE LEGS
    Cube(0.1, 0.1, 2, (-6, -4, 0), (.41, .35, .2), False)
    Cube(0.1, 0.1, 2, (-6, 4, 0), (.41, .35, .2), False)
    Cube(0.1, 0.1, 2, (6, 4, 0), (.41, .35, .2), False)
    Cube(0.1, 0.1, 2, (6, -4, 0), (.41, .35, .2), False)
    
    #TABLE
    Cube(9, 5, 0.25, (0, 0, 2), (0, 0, 1), False)
    Cube(9, 5, 0.01, (0, 0, 2.25), (0, 0, 1), True)
    Cube(9, 0.01, 0.01, (0, 0, 2.25), (0, 0, 1), True)


    #NET
    Cube(0.1, 5, 0.5, (0, 0, 3), (.44, .43, .40), True)

    #BALL
    glPushMatrix()
    quad = gluNewQuadric()
    glTranslate(ballX, ballY, ballZ)
    ball = gluSphere(quad, 0.25, 50, 50)
    glPopMatrix()

    pygame.display.flip()

def point_ended():
    #CURVE CALCULATION
    curves = []

    if(len(bounceLocs) >= 1):

        try:
            pre_bounce_points = ballLocations1[0 : bounceLocs[0].index]
            cstart = getCurve(pre_bounce_points)
            if(cstart != None):
                curves.append(cstart)
        except:
            pass

    
        for i in range(len(bounceLocs) - 1):
            try:
                start_point  = bounceLocs[i].index
                end_point = bounceLocs[i + 1].index
                cmid = getCurve(ballLocations1[start_point : end_point])
                if(cmid != None):
                    curves.append(cmid)
            except:
                pass

        
        try:
            post_bounce_points = ballLocations1[bounceLocs[-1].index : len(ballLocations1)]
            cfinal = getCurve(post_bounce_points)

            if(cfinal != None):
                curves.append(cfinal)
        except:
            pass

    
    turns.append(turn(curves, bounceLocs, ballLocations1))
    bounceLocs.clear()
    ballLocations1.clear()

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

    circles = None
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=1000, param1=cv2.getTrackbarPos('param1', 'sliders'), param2=cv2.getTrackbarPos('param2', 'sliders'), 
    minRadius = cv2.getTrackbarPos('min radius', 'sliders'), maxRadius=cv2.getTrackbarPos('max radius', 'sliders'))


    tableLoc = cv2.getTrackbarPos('Table', 'sliders')
    netLoc = cv2.getTrackbarPos('Net', 'sliders')
    ignore_line = cv2.getTrackbarPos('Ignore', 'sliders')

    cv2.line(img, (0, tableLoc), (width, tableLoc), (0, 255, 0), 3)
    cv2.line(img, (netLoc, 0), (netLoc, width), (0, 0, 255), 3)
    cv2.line(img, (0, ignore_line), (width, ignore_line), (0, 100, 0), 3)

    if circles is not None:
        for i in circles[0]:
            
            x = int(i[0])
            y = int(i[1])
            r = int(i[2])

            if(y > ignore_line):
                break
            
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
                
            if(y >= (tableLoc - 10) and has_moved_above_line):
                if(bounces_on_current_side < 2):
                    bounceLocs.append(loc)
                    first_of_turn = False
                    boing.play()
                    has_moved_above_line = False
                    bounces_on_current_side += 1
                
        
            if(y < (tableLoc - 10)):
                has_moved_above_line = True
            
            if(bounces_on_current_side == 2):
                if(bounceLocs[-1].side == "Left"):
                    right_score += 1
                else:
                    left_score += 1
                
                point_ended()
                beep.play()
                bounces_on_current_side = 0
                first_of_turn = True
                time.sleep(3)

            #cv2.putText(img, f"X: {x} Y: {y} radius: {r}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            #draw Y Line
            cv2.line(img, (x, 0), (x, width), (255, 0, 0), 3)

            #draw X Line
            cv2.line(img, (0, y), (width, y), (255, 0, 0), 3)

            # draw the outer circle
            cv2.circle(img, (x, y), int(i[2]), (0, 0, 255), 2)
            
            # draw the center of the circle
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    
    elif(not first_of_turn):
        if(len(ballLocations1) >= 2):
            
            if(time.time() - ballLocations1[-1].time >= 1.5):
                if(bounces_on_current_side == 0):
                    if(bounceLocs[-1].side == "Left"):
                        right_score += 1
                    else:
                        left_score += 1
                else:
                    if(bounceLocs[-1].side == "Left"):
                        left_score += 1
                    else:
                        right_score += 1
                    
                point_ended()
                beep.play()
                bounces_on_current_side = 0
                first_of_turn = True
                time.sleep(3)


    if(DISPLAY_FPS):
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        fps = str(fps)
        cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    
    cv2.putText(img, f"Left Score: {left_score} Right Score: {right_score}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.imshow('Ball Tracker', img)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        capture.release()
        cv2.destroyAllWindows()
        break


#SHOW CURVE ON GRAPH
# curves = turns[-1].curves
# for parab in curves:
#     x_vals = np.arange(0, width, 1)
#     y_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
#     plt.plot(x_vals, y_vals, 'b')
#     plt.ylim(ymin=invert_y(tableLoc), ymax=height)

# x_points = []
# y_points = []

# for loc in ballLocations1:
#     x_points.append(loc.x)
#     y_points.append(invert_y(loc.y))

# plt.plot(x_points, y_points, 'ok')
# plt.vlines(x=netLoc, ymin=0, ymax=height, color='r', linestyle='-')
# plt.show()

#3D TABLE
pygame.init()

display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
gluPerspective(35, (16 / 9), 0.1, 60.0)
glTranslatef(0.0, 0.0, -30)

glRotatef(-70, 1, 0, 0)
# glRotatef(-45, 0, 0, 1)

ballX = 0
ballY = 0
ballZ = 0

real_table_height = invert_y(tableLoc)

while True:

    for parab in turns[-1].curves:
        x_vals = np.arange(0, width, 1)
        z_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
        scaled_x_vals = (x_vals * 9.0) / float(width / 2)
        scaled_z_vals = (z_vals * 3.0) / float(tableLoc)
    

        #t = parab.time_span
        #wait_per_ball_move = int(t / len(scaled_x_vals))
        wait_per_ball_move = 2

        scaled_intercept = int((parab.intercept * 9.0) / float(width / 2))

        
        for i in range(parab.start, parab.intercept):
            ballX = scaled_x_vals[i] - 9
            ballZ = scaled_z_vals[i]


            update_3D_render(ballX, ballY, ballZ)
            pygame.time.wait(wait_per_ball_move)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit
                    quit()
        
        


    



    



