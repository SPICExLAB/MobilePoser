import math
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

from mobileposer.config import *


DEVICE_POSITIONS = [(-5, 0, -10.0), (-2.5, 0, -10.0), (0, 0, -10.0), (2.5, 0, -10.0), (5, 0, -10.0)]


def init_pygame():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def draw_text(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


def draw_cuboid(w=2, h=2, d=0.4, colors=None):
    w = w / 2
    h = h / 2
    d = d / 2

    colors = [(0.0, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

    glBegin(GL_QUADS)
    glColor3f(*colors[0])

    glVertex3f(w, d, -h)
    glVertex3f(-w, d, -h)
    glVertex3f(-w, d, h)
    glVertex3f(w, d, h)

    glColor3f(*colors[1])

    glVertex3f(w, -d, h)
    glVertex3f(-w, -d, h)
    glVertex3f(-w, -d, -h)
    glVertex3f(w, -d, -h)

    glColor3f(*colors[2])

    glVertex3f(w, d, h)
    glVertex3f(-w, d, h)
    glVertex3f(-w, -d, h)
    glVertex3f(w, -d, h)

    glColor3f(*colors[3])

    glVertex3f(w, -d, -h)
    glVertex3f(-w, -d, -h)
    glVertex3f(-w, d, -h)
    glVertex3f(w, d, -h)

    glColor3f(*colors[4])

    glVertex3f(-w, d, h)
    glVertex3f(-w, d, -h)
    glVertex3f(-w, -d, -h)
    glVertex3f(-w, -d, h)

    glColor3f(*colors[5])

    glVertex3f(w, d, -h)
    glVertex3f(w, d, h)
    glVertex3f(w, -d, h)
    glVertex3f(w, -d, -h)

    glEnd()


def draw(device_id, ori, acc):
    [nx, ny, nz, w] = list(ori)
    glLoadIdentity()
    device_pos = DEVICE_POSITIONS[device_id] 

    glTranslatef(*device_pos)
    draw_text((-0.7, 1.8, 0), list(sensor.device_ids.keys())[device_id], 14)
    glRotatef(2 * math.acos(w) * 180.00/math.pi, nx, nz, ny)
    draw_cuboid(1, 1, 1)


def resize_win(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-7, 7, -7, 7, 0, 15)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()