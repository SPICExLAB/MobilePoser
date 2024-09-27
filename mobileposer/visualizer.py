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
from collections import deque


PLOT = False


class Plot:
    def __init__(self, width, height, max_points=100, y_range=(-1.0, 1.0)):
        """Initialize the plot for visualizing acceleration values."""
        self.width = width
        self.height = height
        self.max_points = max_points
        self.acc_data_x = deque([0] * max_points, maxlen=max_points)
        self.acc_data_y = deque([0] * max_points, maxlen=max_points)
        self.acc_data_z = deque([0] * max_points, maxlen=max_points)
        self.y_range = y_range  # Set the range of the Y-axis (e.g., -1.0 to 1.0)

    def update(self, acc):
        """Update the plot with new acceleration data."""
        acc_x, acc_y, acc_z = acc
        self.acc_data_x.append(acc_x)
        self.acc_data_y.append(acc_y)
        self.acc_data_z.append(acc_z)

    def draw(self):
        """Draw the acceleration plot and add axis scales."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # draw background for the plot
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glVertex2f(self.width, self.height)
        glVertex2f(0, self.height)
        glEnd()

        # draw Y-axis
        self.draw_axis()

        # Draw X, Y, Z acceleration lines
        self.draw_line(self.acc_data_x, (1.0, 0.0, 0.0))  # Red for X
        self.draw_line(self.acc_data_y, (0.0, 1.0, 0.0))  # Green for Y
        self.draw_line(self.acc_data_z, (0.0, 0.0, 1.0))  # Blue for Z

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_line(self, data, color):
        """Draw a line representing one axis of the acceleration data."""
        glColor3f(*color)
        glBegin(GL_LINE_STRIP)
        for i, value in enumerate(data):
            # Scale the Y value to fit the plot's height
            scaled_value = self.scale_value(value, self.y_range[0], self.y_range[1], 0, self.height)
            glVertex2f(i * (self.width / self.max_points), scaled_value)
        glEnd()

    def scale_value(self, value, src_min, src_max, dest_min, dest_max):
        """Scale a value from the source range to the destination range."""
        return dest_min + (float(value - src_min) / (src_max - src_min)) * (dest_max - dest_min)

    def draw_axis(self):
        """Draw Y-axis with tick marks and labels."""
        # Draw Y-axis line
        glColor3f(1.0, 1.0, 1.0)  # White color for axis
        glBegin(GL_LINES)
        glVertex2f(30, 0)  # X = 30 for the Y-axis position
        glVertex2f(30, self.height)
        glEnd()

        # Draw tick marks and labels on the Y-axis
        num_ticks = 5
        tick_spacing = (self.y_range[1] - self.y_range[0]) / num_ticks
        for i in range(num_ticks + 1):
            y_value = self.y_range[0] + i * tick_spacing
            y_pos = self.scale_value(y_value, self.y_range[0], self.y_range[1], 0, self.height)

            # Draw the tick mark
            glBegin(GL_LINES)
            glVertex2f(25, y_pos)  # Tick to the left of Y-axis
            glVertex2f(30, y_pos)
            glEnd()

            # Render the label (Y-value)
            self.draw_text((5, y_pos - 10), f"{y_value:.1f}", 12)

    def draw_text(self, position, textString, size):
        """Draw text at the specified position."""
        font = pygame.font.SysFont("Courier", size, True)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        glRasterPos2d(*position)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


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
        if PLOT:
            self.plot = Plot(width, 200)  # Create a plot for acceleration data
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
        
        # draw cubes
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for cube in self.cubes:
            cube.draw()

        if PLOT:
            # draw the acceleration plot
            glViewport(0, 0, self.width, 200)
            self.plot.draw()
            glViewport(0, 200, self.width, self.height - 200)

        # update the screen
        pygame.display.flip()

    def update(self, acc):
        """Update the screen with new data and render cubes."""
        self.clock.tick(60)
        if PLOT:
            if acc is not None:
                self.plot.update(acc) # draw acceleration graph
        self.draw_scene() # draw cubes  
