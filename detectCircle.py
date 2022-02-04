import cv2
from cv2 import getTrackbarPos
from matplotlib.pyplot import get
import numpy as np
import time
import pygame
from scipy.optimize import curve_fit

capture = cv2.VideoCapture("C:/Users/gavin/Documents/pingpong.mp4")
#capture = cv2.VideoCapture(0)
capture.read()

#capture.set(3, 1280)
#capture.set(4, 720)

if(not capture.isOpened()):
    print("ERROR")


def nothing(a):
    pass

BALL_COLOR = "GREEN"
DISPLAY_FPS = False

cv2.namedWindow('sliders', cv2.WINDOW_NORMAL)
if(BALL_COLOR == "ORANGE"):
    cv2.createTrackbar('param1', 'sliders', 44, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 18, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 18, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 133, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 0, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 137, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 167, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 66, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 255, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 255, 255, nothing)
elif(BALL_COLOR == "WHITE"):
    cv2.createTrackbar('param1', 'sliders', 46, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 13, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 3, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 80, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 32, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 65, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 132, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 97, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 255, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 255, 255, nothing)
elif(BALL_COLOR == "GREEN"):
    cv2.createTrackbar('param1', 'sliders', 46, 200, nothing)
    cv2.createTrackbar('param2', 'sliders', 6, 200, nothing)
    cv2.createTrackbar('min radius', 'sliders', 3, 200, nothing)
    cv2.createTrackbar('max radius', 'sliders', 12, 500, nothing)
    cv2.createTrackbar('R1', 'sliders', 35, 255, nothing)
    cv2.createTrackbar('G1', 'sliders', 26, 255, nothing)
    cv2.createTrackbar('B1', 'sliders', 142, 255, nothing)
    cv2.createTrackbar('R2', 'sliders', 82, 255, nothing)
    cv2.createTrackbar('G2', 'sliders', 146, 255, nothing)
    cv2.createTrackbar('B2', 'sliders', 255, 255, nothing)


cv2.createTrackbar('TableX1', 'sliders', 207, 1000, nothing)
cv2.createTrackbar('TableY1', 'sliders', 263, 1000, nothing)
cv2.createTrackbar('TableX2', 'sliders', 825, 1000, nothing)
cv2.createTrackbar('TableY2', 'sliders', 273, 1000, nothing)


prev_frame_time = 0
new_frame_time = 0

ballLocations = []

height = 0
width = 0

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    
    
    #IF USING PRE-RECORDED VIDEO#
    percent = 50
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    height, width = img.shape[:2]
    #############################
    
    
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

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=cv2.getTrackbarPos('param1', 'sliders'), param2=cv2.getTrackbarPos('param2', 'sliders'), 
    minRadius=cv2.getTrackbarPos('min radius', 'sliders'), maxRadius=cv2.getTrackbarPos('max radius', 'sliders'))


    tableX1 = cv2.getTrackbarPos('TableX1', 'sliders')
    tableY1 = cv2.getTrackbarPos('TableY1', 'sliders')
    tableX2 = cv2.getTrackbarPos('TableX2', 'sliders')
    tableY2 = cv2.getTrackbarPos('TableY2', 'sliders')

    #cv2.line(img, (tableX1, tableY1), (tableX2, tableY2), (255, 0, 0), 3)

    if circles is not None:
        for i in circles[0]:
            
            x = int(i[0])
            y = int(i[1])
            r = int(i[2])

            ballLocations.append([x, y, r, time.time()])
            cv2.putText(img, f"X: {x} Y: {y} radius: {r}", (7, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            #draw Y Line
            cv2.line(img, (x, 0), (x, width), (255, 0, 0), 3)

            #draw X Line
            cv2.line(img, (0, y), (width, y), (0, 255, 0), 3)

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
    
    cv2.imshow('circles', img)
    
    if(cv2.waitKey(1) & 0xFF == ord('a')):
        2
        break

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        capture.release()
        cv2.destroyAllWindows()
        break

pygame.init()
screen = pygame.display.set_mode((width, height))
screen.fill((0, 0, 255))
running = True

startTime = time.time()

while running:
    
    for x, y, r, t in ballLocations:
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 3)
        pygame.display.update()

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


