import os
import time 
import socket
import threading
import math
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import argparse
import select
from argparse import ArgumentParser

from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from numpy.linalg import inv


class Cube:
    def __init__(self, device_id, position):
        self.device_id = device_id
        self.position = position
        self.orientation = [0, 0, 0, 1]  # Quaternion (nx, ny, nz, w)
        self.size = (1, 1, 1)

    def set_orientation(self, ori):
        """Update the cube's orientation based on the received data."""
        self.orientation = ori

    def draw(self):
        """Draw the cube at its current position with the specified orientation."""
        [nx, ny, nz, w] = list(self.orientation)
        glLoadIdentity()
        glTranslatef(*self.position)
        self.draw_text((-0.7, 1.8, 0), f"Device {self.device_id}", 14)
        angle = 2 * math.acos(w) * 180.00 / math.pi
        glRotatef(angle, nx, nz, ny)
        self.draw_cuboid(self.size[0], self.size[1], self.size[2])

    def draw_text(self, position, textString, size):
        """Draw text at the specified position."""
        font = pygame.font.SysFont("Courier", size, True)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        glRasterPos3d(*position)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

    def draw_cuboid(self, w=1, h=1, d=1):
        """Draw a cuboid (cube) with the given width, height, and depth."""
        w, h, d = w / 2, h / 2, d / 2
        colors = [(0.0, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

        glBegin(GL_QUADS)
        for i, color in enumerate(colors):
            glColor3f(*color)
            if i == 0:
                glVertex3f(w, d, -h)
                glVertex3f(-w, d, -h)
                glVertex3f(-w, d, h)
                glVertex3f(w, d, h)
            elif i == 1:
                glVertex3f(w, -d, h)
                glVertex3f(-w, -d, h)
                glVertex3f(-w, -d, -h)
                glVertex3f(w, -d, -h)
            elif i == 2:
                glVertex3f(w, d, h)
                glVertex3f(-w, d, h)
                glVertex3f(-w, -d, h)
                glVertex3f(w, -d, h)
            elif i == 3:
                glVertex3f(w, -d, -h)
                glVertex3f(-w, -d, -h)
                glVertex3f(-w, d, -h)
                glVertex3f(w, d, -h)
            elif i == 4:
                glVertex3f(-w, d, h)
                glVertex3f(-w, d, -h)
                glVertex3f(-w, -d, -h)
                glVertex3f(-w, -d, h)
            elif i == 5:
                glVertex3f(w, d, -h)
                glVertex3f(w, d, h)
                glVertex3f(w, -d, h)
                glVertex3f(w, -d, -h)
        glEnd()


class PyGameManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cubes = []  # List to hold Cube objects
        self.clock = pygame.time.Clock()
        self.init_pygame()

    def init_pygame(self):
        """Initialize PyGame and OpenGL settings."""
        video_flags = OPENGL | DOUBLEBUF
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), video_flags)
        pygame.display.set_caption("IMU orientation visualization")
        self.resize_win(self.width, self.height)
        self.setup_opengl()

    def setup_opengl(self):
        """Configure OpenGL settings."""
        glShadeModel(GL_SMOOTH)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    def resize_win(self, width, height):
        """Resize the OpenGL viewport when the window size changes."""
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-7, 7, -7, 7, 0, 15)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def add_cube(self, cube):
        """Add a cube to the PyGame manager."""
        self.cubes.append(cube)

    def handle_events(self):
        """Return whether to continue running and if 'c' was pressed."""
        key_c_pressed = False
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False, key_c_pressed
            if event.type == KEYUP and event.key == K_c:
                key_c_pressed = True
        return True, key_c_pressed

    def draw_scene(self):
        """Clear the screen and draw all the cubes."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for cube in self.cubes:
            cube.draw()
        pygame.display.flip()

    def update(self):
        """Update the screen with new data and render cubes."""
        self.clock.tick(0)
        self.draw_scene()
