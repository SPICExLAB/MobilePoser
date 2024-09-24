import os
import time
import threading
import queue
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from argparse import ArgumentParser
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

from mobileposer.config import *
from mobileposer.utils.sensor_utils import *
from mobileposer.utils.socket_utils import *
from mobileposer.utils.draw_utils import *
from mobileposer.cube import *
from mobileposer.teensy import TeensySerialManager

# Initialize a queue for sensor data
sensor_queue = queue.Queue()

def read_sensor_data(teensy, queue):
    """Continuously queue sensor data."""
    while True:
        try:
            teensy.serial_read_val()
            recent_imu = np.array(teensy.get_recent_data_imu())[0]
            timestamps, imu = recent_imu
            ori = np.array(imu[9:13])
            queue.put((0, ori))  
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    # Start Teensy Serial Manager
    teensy = TeensySerialManager(data_dict={}, numElect=8)
    teensy.serial_start()

    # Setup PyGame manager
    manager = PyGameManager(860, 860)

    # Initialize cube for the device and add it to the manager
    cube = Cube(device_id=0, position=DEVICE_POSITIONS[0])
    manager.add_cube(cube)

    # Start the sensor reading thread
    sensor_thread = threading.Thread(target=read_sensor_data, args=(teensy, sensor_queue))
    sensor_thread.daemon = True
    sensor_thread.start()

    clock = pygame.time.Clock()

    while True:
        # Handle pygame events
        running, _ = manager.handle_events()
        if not running:
            break

        # Process  available sensor data
        while not sensor_queue.empty():
            device_id, ori = sensor_queue.get()
            cube.set_orientation(ori)

        # Update and render
        manager.update()

        # Limit the frame rate
        clock.tick(60)

    # Cleanup
    teensy.serial_stop()
    pygame.quit()