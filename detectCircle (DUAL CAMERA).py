from math import sqrt
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
capture2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

width = 640
height = 480

#RESIZES LIVE CAMERA FEED
capture.set(3, width)
capture.set(4, height)
capture2.set(3, width)
capture2.set(4, height)

assert capture.isOpened()
assert capture2.isOpened()

#NEEDED FOR SLIDERS TO WORK
def nothing(a):
    pass

BALL_COLOR = "PINK"

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
    cv2.createTrackbar('param2', 'sliders', 1, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 1, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 50, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 125, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 106, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 130, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 171, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 238, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 248, 255, nothing)

cv2.createTrackbar('Table', 'sliders', 193, width, nothing)

cv2.createTrackbar('Net', 'sliders', 310, width, nothing)

cv2.createTrackbar('Ignore', 'sliders', 260, width, nothing )

prev_frame_time = 0
new_frame_time = 0

ballLocations1 = []
ballLocations2 = []

locCount = 0

bounceLocs = []
turns = []
bounces_on_current_side = 0
has_moved_above_line = True

left_score = 0
right_score = 0

first_of_turn = True

bounces_3d = []

VERTICAL_ROTATION = -65
HORIZONTAL_ROTATION = -30

mixer.init()
beep = pygame.mixer.Sound('./beep.ogg')
boing = pygame.mixer.Sound("./boing.ogg")

def invert_y(y):
    return height - y

def quadratic(x, a, b, c):
    y = a * (x - b) * (x - b) + c
    return y

def solve_quadratic(y, a, b, c, direction):
    if(direction == "Right"):
        return int(sqrt((y - c) / a) + b)
    
    return int(-sqrt((y - c) / a) + b)
      
def getCurve(points, off_table):

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

    delta_time = points[-1].time - points[0].time
    
    direction = "ERROR"
    
    if(points[0].x < points[-1].x):
        direction = "Right"
    else:
        direction = "Left"

    table_intercept = solve_quadratic(invert_y(tableLoc), a, b, c, direction)
    
    return curve(a, b, c, x[0], table_intercept, delta_time, off_table, direction)

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

    for s in surfaces[1:]:
        glColor3fv(color)
        for vertex in s:
            glVertex3fv(verticies[vertex])
    
    glEnd()

def draw_path(points):
    vert_list = tuple(points)
    temp  = []
    
    for i in range(len(vert_list) - 1):
        temp.append((i, i + 1))
    
    edges = tuple(temp)

    glLineWidth(4)

    glBegin(GL_LINES)
    
    for edge in edges:
        for vertex in edge:
            glColor3fv((1, 0, 0))
            glVertex3fv(vert_list[vertex])
            
    glEnd()
    
def update_3D_render(ballX, ballY, ballZ, path_points):
    glLineWidth(1)
    # glRotatef(0.05, 0, 0, 1)
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

    #BOUNCE SPOTS
    for pos in bounces_3d:
        x, y, z = pos
        
        Circle(0.25, 50, (x, y, 2.25), (0.2, 0.2, 0.2), False)
    
    if(path_points != None):
        draw_path(path_points)
    
    pygame.display.flip()

def point_ended(double_bounce):
    #CURVE CALCULATION
    curves = []

    if(len(bounceLocs) >= 1):

        try:
            pre_bounce_points = ballLocations1[0 : bounceLocs[0].index]
            cstart = getCurve(pre_bounce_points, False)
            if(cstart != None):
                curves.append(cstart)
        except:
            pass

    
        for i in range(len(bounceLocs)):
            try:
                start_point  = bounceLocs[i].index
                end_point = bounceLocs[i + 1].index
                cmid = getCurve(ballLocations1[start_point : end_point], False)
                if(cmid != None):
                    curves.append(cmid)
            except:
                pass

        
        try:
            post_bounce_points = ballLocations1[bounceLocs[-1].index : len(ballLocations1)]
            if(double_bounce):
                cfinal = getCurve(post_bounce_points, False)
            else:
               cfinal = getCurve(post_bounce_points, True) 

            if(cfinal != None):
                curves.append(cfinal)
        except:
            pass

    
    turns.append(turn(curves, bounceLocs, ballLocations1))
    bounceLocs.clear()
    ballLocations1.clear()

#TAKEN DIRECTLY FROM STACK OVERFLOW
def drawText(x, y, text, size):   
    font = pygame.font.SysFont('freesansbold.tff', size)                                             
    textSurface = font.render(text, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def getCircle(img):
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

    return circles

counter = 0
frame_sum = 0
frame_count = 0

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
    
    circles = getCircle(img)
    img2 = None

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

            loc = ballLoc(x, y, r, time.time(), side, locCount)
            ballLocations1.append(loc)
            locCount += 1

            circles2 = None
            
            if(counter % 2 == 0):
                _, img2 = capture2.read()
                img2 = cv2.flip(img2, 1)
                
                circles2 = getCircle(img2)

                if circles2 is not None:
                    for i in circles2[0]:
                        
                        x2 = int(i[0])
                        y2 = int(i[1])
                        r2 = int(i[2])

                        cv2.line(img2, (x2, 0), (x2, width), (255, 0, 0), 3)
                        cv2.line(img2, (0, y2), (width, y2), (255, 0, 0), 3)
                        cv2.circle(img2, (x2, y2), int(r2), (0, 0, 255), 2)
                        cv2.circle(img2, (x2, y2), 2, (0, 0, 255), 3)

                        loc2 = ballLoc(x2, y2, r2, time.time(), side, locCount)
                        ballLocations2.append(loc2)
                else:
                    ballLocations2.append(None)

            #draw Y Line
            cv2.line(img, (x, 0), (x, width), (255, 0, 0), 3)

            #draw X Line
            cv2.line(img, (0, y), (width, y), (255, 0, 0), 3)

            # draw the outer circle
            cv2.circle(img, (x, y), int(r), (0, 0, 255), 2)
            
            # draw the center of the circle
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
            
            if(len(ballLocations1) >= 2):
                if(ballLocations1[-2].side != ballLocations1[-1].side):
                    bounces_on_current_side = 0
                
            if(y >= (tableLoc) and has_moved_above_line):
                if(bounces_on_current_side < 2):
                    bounceLocs.append(loc)
                    first_of_turn = False
                    boing.play()
                    has_moved_above_line = False
                    bounces_on_current_side += 1
                
        
            if(y < (tableLoc)):
                has_moved_above_line = True
            
            if(bounces_on_current_side == 2):
                if(bounceLocs[-1].side == "Left"):
                    right_score += 1
                else:
                    left_score += 1
                
                bounceLocs.append(loc)
                point_ended(True)
                beep.play()
                bounces_on_current_side = 0
                first_of_turn = True
                locCount = 0
                time.sleep(3)

            #cv2.putText(img, f"X: {x} Y: {y} radius: {r}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

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
                        right_score += 1
                    else:
                        left_score += 1
                    
                point_ended(False)
                beep.play()
                bounces_on_current_side = 0
                first_of_turn = True
                locCount = 0
                time.sleep(3)

    DISPLAY_FPS = True
    if(DISPLAY_FPS):
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        frame_sum += fps
        frame_count += 1
        avg = int(frame_sum / frame_count)
        avg = str(avg)
        cv2.putText(img, avg, (10, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        if(frame_count == 25):
            frame_count = 0
            frame_sum = 0

    
    cv2.putText(img, f"Left Score: {left_score} Right Score: {right_score}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.imshow('Horiz Camera', img)
    if img2 is not None:
        cv2.imshow('Vert Camera', img2)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        capture.release()
        cv2.destroyAllWindows()
        break
    
    counter += 1


show_curve = False

if(show_curve):
    #SHOW CURVE ON GRAPH
    curves = turns[-1].curves
    for parab in curves:
        x_vals = np.arange(0, width, 1)
        y_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
        plt.plot(x_vals, y_vals, 'b')
        plt.ylim(ymin=invert_y(tableLoc), ymax=height)

    # x_points = []
    # y_points = []

    # for loc in ballLocations1:
    #     x_points.append(loc.x)
    #     y_points.append(invert_y(loc.y))

    # plt.plot(x_points, y_points, 'ok')
    plt.vlines(x=netLoc, ymin=0, ymax=height, color='r', linestyle='-')
    plt.show()

#3D TABLE
pygame.init()

display = (800, 600)
screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
gluPerspective(35, (16 / 9), 0.1, 60.0)
glTranslatef(0.0, 0.0, -30)

glRotatef(VERTICAL_ROTATION, 1, 0, 0)
glRotatef(HORIZONTAL_ROTATION, 0, 0, 1)

ballX = 99
ballY = 0
ballZ = 99

real_table_height = invert_y(tableLoc)

points = []
PATH_LENGTH = 80
prev_end = 0

for parab in turns[-1].curves:
    end = parab.intercept
    if(not parab.off_table and parab.intercept > width):
        end = width - 1

    x_vals = np.arange(0, width, 1)

    z_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
    scaled_x_vals = (x_vals * 18.0) / float(width)
    scaled_z_vals = (z_vals * 3.0) / float(real_table_height)

    coefficient = 5

    t = parab.time_span
    wait_per_ball_move = t / (len(scaled_x_vals) / coefficient)

    scaled_intercept = int((parab.intercept * 18.0) / float(width)) - 9

    if(parab.direction == "Right"):
        for i in range(prev_end, end, coefficient):
            if(i > width):
                break

            ballX = scaled_x_vals[i] - 9
            ballZ = scaled_z_vals[i] - 0.7
            points.append((ballX, 0, ballZ))

            update_3D_render(ballX, ballY, ballZ, points)
            time.sleep(wait_per_ball_move)

            if(len(points) % PATH_LENGTH == 0):
                points.pop(0)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit
                    quit()
    
    elif(parab.direction == "Left"):
        if(prev_end == 0):
            prev_end = width - 1

        for i in range(prev_end, end, -coefficient):
            ballX = scaled_x_vals[i] - 9
            ballZ = scaled_z_vals[i] - 0.7
            points.append((ballX, 0, ballZ))

            update_3D_render(ballX, ballY, ballZ, points)
            time.sleep(wait_per_ball_move)

            if(len(points) % PATH_LENGTH == 0):
                points.pop(0)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit
                    quit()

    bounces_3d.append((ballX, ballY, 2.1))
    prev_end = parab.intercept
    
update_3D_render(0, 0, 100, None)

NUM_FRAMES = 75
per_frame_vertical = -float(VERTICAL_ROTATION) / NUM_FRAMES
per_frame_horizontal = -float(HORIZONTAL_ROTATION) / NUM_FRAMES

for i in range(NUM_FRAMES):
    glRotatef(per_frame_horizontal, 0, 0, 1)
    update_3D_render(0, 0, 100, None)

for i in range(NUM_FRAMES):
    glRotatef(per_frame_vertical, 1, 0, 0)
    update_3D_render(0, 0, 100, None)

speed_sum = 0
tot = 0
for c in turns[-1].curves:
    speed_sum += (abs(c.intercept - c.start) / c.time_span)
    tot += 1

avg_speed = int(((speed_sum / tot) / (width / 9)) * 14.67) / 10.0

drawText(50, 50, f"Average Speed: {avg_speed} mph", 45)
pygame.display.flip()

time.sleep(5)
pygame.quit()
quit()