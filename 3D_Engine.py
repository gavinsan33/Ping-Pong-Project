import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import * 
import numpy as np

def Pyr():
    pyr_vert = (

    (0, 1, 0),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, -1, -1),
    (1, -1, -1),
    
    )

    pyr_edges = (

        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),

    )

    pyr_surfaces = (

        (1, 2, 4, 3),
        (0, 1, 2),
        (0, 2, 4),
        (0, 3, 4),
        (0, 1, 3),

    )

    glBegin(GL_QUADS)
    glColor3fv((1, 0, 0))
    
    for vertex in pyr_surfaces[0]:
        glVertex3fv(pyr_vert[vertex])

    glEnd()

    glBegin(GL_TRIANGLES)
    
    for s in pyr_surfaces[1:]:
        for vertex in s:
            glVertex3fv(pyr_vert[vertex])

    glEnd()
    
    glBegin(GL_LINES)

    for edge in pyr_edges:
        for vertex in edge:
            glVertex3fv(pyr_vert[vertex])
            
    glEnd()

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

pygame.init()

display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
gluPerspective(35, (16 / 9), 0.1, 60.0)
glTranslatef(0.0, 0.0, -30)

VERTICAL_ROTATION = -65
HORIZONTAL_ROTATION = -30

glRotatef(VERTICAL_ROTATION, 1, 0, 0)
glRotatef(HORIZONTAL_ROTATION, 0, 0, 1)

x = -5.0
y = -2.0

NUM_FRAMES = 300
per_frame_vertical = -float(VERTICAL_ROTATION) / NUM_FRAMES
per_frame_horizontal = -float(HORIZONTAL_ROTATION) / NUM_FRAMES

while True:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit
            quit()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glRotatef(per_frame_horizontal, 0, 0, 1)
    # glRotatef(per_frame_vertical, 1, 0, 0)

    #FLOOR
    #Cube(20, 20, 0.01, (0, 0, 0), (0, 1, 1), False)
    Circle(15, 500, (0, 0, 0), (0, 1, 0.25), False)

    #TABLE LEGS
    Cube(0.1, 0.1, 2, (-6, -4, 0), (.41, .35, .2), False)
    Cube(0.1, 0.1, 2, (-6, 4, 0), (.41, .35, .2), False)
    Cube(0.1, 0.1, 2, (6, 4, 0), (.41, .35, .2), False)
    Cube(0.1, 0.1, 2, (6, -4, 0), (.41, .35, .2), False)
    
    #TABLE
    Cube(9, 5, 0.25, (0, 0, 2), (0, 0, 1), False)
    Cube(9, 5, 0.01, (0, 0, 2.25), (0, 0, 1), True)
    Cube(9, 0.01, 0.01, (0, 0, 2.25), (0, 0, 1), True)

    #BALL
    glPushMatrix()
    quad = gluNewQuadric()
    glTranslate(x, y, 2.5)
    ball = gluSphere(quad, 0.25, 50, 50)
    x += 0.1
    y += 0.0
    glPopMatrix()

    #NET
    Cube(0.1, 5, 0.5, (0, 0, 3), (.44, .43, .40), True)


    pygame.display.flip()
    pygame.time.wait(10)
    