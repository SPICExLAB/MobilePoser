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

from mobileposer.config import *
from mobileposer.utils.sensor_utils import *
from mobileposer.utils.socket_utils import * 
from mobileposer.utils.draw_utils import *
from mobileposer.visualizer import *


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    # Init socket and data handlers
    sockets = init_sockets(HOST, PORTS)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sensor_data = SensorData()

    # Setup PyGame manager
    manager = PyGameManager(860, 860)

    # Initialize cubes for each device 
    cubes = [Cube(device_id=i, position=DEVICE_POSITIONS[i]) for i in range(len(DEVICE_POSITIONS))]
    for cube in cubes:
        manager.add_cube(cube)

    frames = 0
    empty = []
    prev_timestamp = 0
    curr_timestamp = 0

    while True:
        continue_running, key_c_pressed = manager.handle_events()
        if not continue_running:
            break

        if key_c_pressed:
            print("Started Calibration!")
            sensor_data.calibrate()

        # read data
        try:
            readable, writable, exceptional = select.select(sockets, empty, empty, 0.001)
            for sock in readable:
                # read data from socket
                data, addr = sock.recvfrom(CHUNK)

                # process received data
                vis_str, device_id, curr_acc, curr_ori, timestamps = process_data(data)
                curr_timestamp = sensor_data.update(device_id, curr_acc, curr_ori, timestamps)
                glb_ori, glb_acc = sensor2global(sensor_data.get_orientation(device_id), 
                                        sensor_data.get_acceleration(device_id), 
                                        sensor_data.calibration_quats, 
                                        device_id)
                sensor_data.update_virtual(device_id, glb_acc, glb_ori) # update virtual data for display

                # Update cube orientation
                cubes[device_id].set_orientation(sensor_data.virtual_ori[device_id])

                time_diff = curr_timestamp - prev_timestamp
                if time_diff >= min_time_diff:
                    # send data via socket to live demo
                    send_and_save_data(send_sock, sensor_data.virtual_acc, sensor_data.virtual_ori)
                    prev_timestamp = curr_timestamp

                # Draw cubes using the PyGameManager
                manager.update(glb_acc)

        except KeyboardInterrupt:
            print("==== Close Socket ====")
            os._exit(0)
        except Exception as e:
            print("Exception encountered: ", e)
