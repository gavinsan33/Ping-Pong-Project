from math import sqrt
from ntpath import join
from re import X
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
from scipy.stats import linregress

#capture = cv2.VideoCapture("./pingpong.mp4")
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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
    cv2.createTrackbar('R1', 'sliders', 62, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 62, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 56, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 94, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 120, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 186, 255, nothing)
elif(BALL_COLOR == "PINK"):
    cv2.createTrackbar('param1', 'sliders', 27, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 5, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 1, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 11, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 135, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 71, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 52, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 172, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 180, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 149, 255, nothing)

cv2.createTrackbar('Table', 'sliders', 183, width, nothing)
cv2.createTrackbar('Net', 'sliders', 335, width, nothing)
cv2.createTrackbar('Ignore', 'sliders', 218, width, nothing)

# cv2.createTrackbar('VertStartL1X1', 'sliders', 0, width, nothing)
# cv2.createTrackbar('VertStartL1Y1', 'sliders', 278, height, nothing)
# cv2.createTrackbar('VertStartL1X2', 'sliders', 640, width, nothing)
# cv2.createTrackbar('VertStartL1Y2', 'sliders', 316, height, nothing)

cv2.createTrackbar('Vert Start', 'sliders', 327, height, nothing)
cv2.createTrackbar('Vert End', 'sliders', 20, height, nothing)
cv2.createTrackbar('ShortL', 'sliders', 79, width, nothing)
cv2.createTrackbar('ShortR', 'sliders', 580, width, nothing)

prev_frame_time = 0
new_frame_time = 0

ballLocations1 = []

locCount = 0

bounceLocs = []
turns = []
bounces_on_current_side = 0
has_moved_above_line = True

has_passed_left_bound = False
has_passed_right_bound = False

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

def linear(m, x, b):
    return m * x + b

def solve_quadratic(y, a, b, c, direction):
    if(direction == "Right"):
        return int(sqrt((y - c) / a) + b)
    
    return int(-sqrt((y - c) / a) + b)
      
def getCurve(points, off_table):

    if(len(points) <= 3):
        return

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

    direction = "ERROR"
    
    mid = int(len(points) / 2)

    med1 = points[int(mid / 2)].x
    med2 = points[mid + int(mid / 2)].x
    
    if(med1 < med2):
        direction = "Right"
    else:
        direction = "Left"

    table_intercept = solve_quadratic(invert_y(tableLoc), a, b, c, direction)
    
    line = getLine(points)

    delta_time = 2 * (points[-1].time - points[mid].time)

    if(a < 0 and r_squared >= 0.5):
        return curve(a, b, c, x[0], table_intercept, delta_time, off_table, direction, line)
    
    new_points = []
    for i in range(1, len(points) - 1):
        new_points.append(points[i])
    
    print("Curve calculation failed... using failsafe")
    return getCurve(new_points, off_table)

def getLine(points):
    
    x_vals = []
    y_vals = []
    
    for p in points:
        if p.depth is not None:
            x_vals.append(p.x)
            y_vals.append(invert_y(p.depth))

    x = np.array(x_vals)
    y = np.array(y_vals)

    slope, intercept, _, _, _ = linregress(x, y)
    return (slope, intercept)

def get_line_from_points(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1
    slope = delta_y / delta_x
    intercept = (-slope * x1) + y1

    return (slope, intercept)

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
    
def update_3D_render(ballX, ballY, ballZ, path_points, text):
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
    
    drawText(50, 50, text, 45)
    drawText(20, height2 - 50, f"Left Score: {str(left_score)}", 50)
    drawText(width2 - 250, height2 - 50, f"Right Score: {str(right_score)}", 50)
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
    bounces_3d.clear()
    has_passed_left_bound = False
    has_passed_right_bound = False
    
    locCount = 0

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

vert_start = cv2.getTrackbarPos('Vert Start', 'sliders')
vert_end = cv2.getTrackbarPos('Vert End', 'sliders')

shortL = cv2.getTrackbarPos('ShortL', 'sliders')
shortR = cv2.getTrackbarPos('ShortR', 'sliders')

tableLoc = cv2.getTrackbarPos('Table', 'sliders')

#3D TABLE
pygame.init()
width2, height2 = 800, 600
display = (width2, height2)
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

##### SHOW REPLAY OF POINT ########
PASS = False
def show_replay():
    global prev_end
    if(PASS):
        return

    show_curve = False

    if(show_curve):

        #SHOW CURVE ON GRAPH
        curves = turns[-1].curves
        for parab in curves:
            x_vals = np.arange(0, width, 1)
            y_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
            line = parab.line
            slope, intercept = line
            y_vals2 = x_vals * slope + intercept
            plt.plot(x_vals, y_vals, 'b')
            plt.plot(x_vals, y_vals2, 'g')
            plt.ylim(ymin=invert_y(tableLoc), ymax=height)

        x_points = []
        y_points = []

        locations = turns[-1].ballLocs    
        for loc in locations:
            x_points.append(loc.x)
            y_points.append(invert_y(loc.y))

        plt.plot(x_points, y_points, 'ok')
        plt.vlines(x=netLoc, ymin=0, ymax=height, color='r', linestyle='-')
        plt.show()

    ballX = 99
    ballY = 0
    ballZ = 99
    prev_end = 0
    points.clear()

    # x1 = cv2.getTrackbarPos('VertStartL1X1', 'sliders')
    # y1 = invert_y(cv2.getTrackbarPos('VertStartL1Y1', 'sliders'))
    # x2 = cv2.getTrackbarPos('VertStartL1X2', 'sliders')
    # y2 = invert_y(cv2.getTrackbarPos('VertStartL1Y2', 'sliders'))

    # slope2, intercept2 = get_line_from_points(x1, y1, x2, y2)
    
    for parab in turns[-1].curves:
        end = parab.intercept
        if(not parab.off_table and parab.intercept > width):
            end = width - 1

        x_vals = np.arange(0, width, 1)

        z_vals = quadratic(x_vals, parab.a, parab.b, parab.c)
        slope, intercept = parab.line
        y_vals = linear(slope, x_vals, intercept)

        scaled_x_vals = (x_vals * 18.0) / float(shortR - shortL) - 11
        scaled_y_vals = (y_vals * 10.0) / float(vert_start - vert_end) - 10
        scaled_z_vals = (z_vals * 3.0) / float(real_table_height)

        t = parab.time_span
        wait_per_ball_move = t / (len(scaled_x_vals) / 2)

        if(parab.direction == "Right"):
            for i in range(prev_end, end, 2):
                if(i > width):
                    break

                ballX = scaled_x_vals[i]
                ballY = scaled_y_vals[i]
                ballZ = scaled_z_vals[i] - 0.7
                points.append((ballX, ballY, ballZ))

                update_3D_render(ballX, ballY, ballZ, points, "")
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

            for i in range(prev_end, end, -2):
                ballX = scaled_x_vals[i] 
                ballY = scaled_y_vals[i]
                ballZ = scaled_z_vals[i] - 0.7
                points.append((ballX, ballY, ballZ))

                update_3D_render(ballX, ballY, ballZ, points, "")
                time.sleep(wait_per_ball_move)

                if(len(points) % PATH_LENGTH == 0):
                    points.pop(0)

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        pygame.quit
                        quit()

        bounces_3d.append((ballX, ballY, 2.1))
        prev_end = parab.intercept
        
    update_3D_render(0, 0, 100, None, "")

    NUM_FRAMES = 75
    per_frame_vertical = -float(VERTICAL_ROTATION) / NUM_FRAMES
    per_frame_horizontal = -float(HORIZONTAL_ROTATION) / NUM_FRAMES

    for i in range(NUM_FRAMES):
        glRotatef(per_frame_horizontal, 0, 0, 1)
        update_3D_render(0, 0, 100, None, "")
        time.sleep(0.01)

    for i in range(NUM_FRAMES):
        glRotatef(per_frame_vertical, 1, 0, 0)
        update_3D_render(0, 0, 100, None, "")
        time.sleep(0.01)

    speed_sum = 0
    tot = 0
    for c in turns[-1].curves:
        speed_sum += (abs(c.intercept - c.start) / c.time_span)
        tot += 1

    avg_speed = 0
    if(tot != 0):
        avg_speed = int(((speed_sum / tot) / (width / 9)) * 14.67) / 10.0

    for i in range(NUM_FRAMES):
        #DO NOTHING
        
        update_3D_render(0, 0, 100, None, f"Average Speed: {avg_speed} mph")
        time.sleep(0.01)
    
    for i in range(NUM_FRAMES):
        glRotatef(-per_frame_vertical, 1, 0, 0)
        update_3D_render(0, 0, 100, None, "")
        time.sleep(0.01)
        
    for i in range(NUM_FRAMES):
        glRotatef(-per_frame_horizontal, 0, 0, 1)
        update_3D_render(0, 0, 100, None, "")
        time.sleep(0.01)

    # pygame.quit()

while True:
    success, img = capture.read()
    # img = cv2.flip(img, 1)
    circles = getCircle(img)

    # #IF USING PRE-RECORDED VIDEO#
    # percent = 50
    # width = int(img.shape[1] * percent / 100)
    # height = int(img.shape[0] * percent / 100)
    # dim = (width, height)

    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # height, width = img.shape[:2]
    # #############################
    
    _, img2 = capture2.read()
    # img2 = cv2.flip(img2, 1)

    tableLoc = cv2.getTrackbarPos('Table', 'sliders')
    netLoc = cv2.getTrackbarPos('Net', 'sliders')
    ignore_line = cv2.getTrackbarPos('Ignore', 'sliders')

    vert_start = cv2.getTrackbarPos('Vert Start', 'sliders')
    vert_end = cv2.getTrackbarPos('Vert End', 'sliders')

    shortL = cv2.getTrackbarPos('ShortL', 'sliders')
    shortR = cv2.getTrackbarPos('ShortR', 'sliders')

    # VertL1X1 = cv2.getTrackbarPos('VertStartL1X1', 'sliders')
    # VertL1Y1 = cv2.getTrackbarPos('VertStartL1Y1', 'sliders')
    # VertL1X2 = cv2.getTrackbarPos('VertStartL1X2', 'sliders')
    # VertL1Y2 = cv2.getTrackbarPos('VertStartL1Y2', 'sliders')
    
    cv2.line(img, (0, tableLoc), (width, tableLoc), (0, 255, 0), 3)
    cv2.line(img2, (netLoc, 0), (netLoc, width), (0, 0, 255), 3)
    cv2.line(img, (0, ignore_line), (width, ignore_line), (0, 100, 0), 3)
    cv2.line(img2, (0, vert_end), (width, vert_end), (0, 255, 0), 3)
    cv2.line(img2, (0, vert_start), (width, vert_start), (0, 255, 0), 3)
    cv2.line(img2, (shortL, 0), (shortL, width), (100, 0, 100), 3)
    cv2.line(img2, (shortR, 0), (shortR, width), (100, 0, 100), 3)
    # cv2.line(img2, (VertL1X1, VertL1Y1), (VertL1X2, VertL1Y2), (0, 255, 0), 3)
    
    if circles is not None:
        
        for i in circles[0]:
            
            x = int(i[0])
            y = int(i[1])
            r = i[2]

            if(y > ignore_line):
                break
            
            side = None

            if(x < netLoc):
                side = "Left"
            else:
                side = "Right"

            loc = ballLoc(y, r, time.time(), side, locCount)
            loc.alt_x = x

            if(counter % 1 == 0):
                circles2 = getCircle(img2)
                if circles2 is not None:
                    for i in circles2[0]:
                        x2 = int(i[0])
                        y2 = int(i[1])
                        r2 = i[2]

                        #draw Y Line
                        cv2.line(img2, (x2, 0), (x2, width), (255, 0, 0), 3)

                        #draw X Line
                        cv2.line(img2, (0, y2), (width, y2), (255, 0, 0), 3)

                        # draw the outer circle
                        cv2.circle(img2, (x2, y2), int(r2), (0, 0, 255), 2)
                        
                        # draw the center of the circle
                        cv2.circle(img2, (x2, y2), 2, (0, 0, 255), 3)

                        if(x2 < shortL):
                            has_passed_left_bound = True
                        else:
                            has_passed_left_bound = False

                        if(x2 > shortR):
                            has_passed_right_bound = True
                        else:
                            has_passed_right_bound = False

                        loc.x = x2
                        loc.depth = y2
            
            if(loc.x != None):
                ballLocations1.append(loc)
                locCount += 1

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
                    
                if(y >= tableLoc and has_moved_above_line):
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
                    show_replay()

            # cv2.putText(img, f"radius: {r}", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    
            

    elif(not first_of_turn):
        if(len(ballLocations1) >= 2):
            if(time.time() - ballLocations1[-1].time >= 1.5):
                #USES BLUE LINES FOR SCORE
                print("No ball found")
                if(has_passed_left_bound or has_passed_right_bound):
                    if(bounces_on_current_side == 0):
                        if(has_passed_left_bound):
                            right_score += 1
                        else:
                            left_score += 1
                    else:
                        if(bounceLocs[-1].side == "Left"):
                            right_score += 1
                        else:
                            left_score += 1
                else:
                    #FAILSAFE WHICH USES BOUNCE SIDE
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
                show_replay()

    DISPLAY_FPS = True
    if(DISPLAY_FPS):
        if(frame_count == 25):
            frame_count = 0
            frame_sum = 0

        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        frame_sum += fps
        frame_count += 1
        avg = int(frame_sum / frame_count)
        avg = str(avg)
        cv2.putText(img, avg, (10, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    #cv2.putText(img, f"Left Score: {left_score} Right Score: {right_score}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.imshow('Horiz Camera', img)
    if img2 is not None:
        cv2.imshow('Vert Camera', img2)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        sys.exit(0)
    
    counter += 1



