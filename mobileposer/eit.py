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
from mobileposer.constants import * 
from mobileposer.utils.sensor_utils import *
from mobileposer.utils.socket_utils import *
from mobileposer.utils.draw_utils import *
from mobileposer.visualizer import *
from mobileposer.teensy import TeensySerialManager

# Initialize a queue for sensor data
sensor_queue = queue.Queue()


def compute_ema(curr_val, prev_val, alpha=0.2):
    """Compute exponential moving average."""
    return alpha * curr_val + (1 - alpha) * prev_val


def read_sensor_data(teensy, queue):
    """Continuously queue sensor data."""
    while True:
        try:
            teensy.serial_read_val()
            recent_imu = np.array(teensy.get_recent_data_imu())[0]
            timestamps, imu = recent_imu
            acc = np.array(imu[:3])
            ori = np.array(imu[9:13])
            queue.put((0, acc, ori))  
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    # Track timing
    prev_timestamp = 0
    curr_timestamp = 0

    # Init data sending socket
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Start Teensy Serial Manager
    teensy = TeensySerialManager(data_dict={}, numElect=8)
    teensy.serial_start()

    # Setup PyGame manager
    manager = PyGameManager(860, 860)

    # Initialize cube for the device and add it to the manager
    sensor_data = SensorData()
    cube = Cube(device_id=0, position=DEVICE_POSITIONS[0])
    manager.add_cube(cube)

    # Track EMA for acceleration 
    ema_acc = None 
    alpha = 0.5

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
            device_id, acc, ori = sensor_queue.get()
            cube.set_orientation(ori)

            # Compute EMA for acceleration
            if ema_acc is None:
                ema_acc = acc  # First value becomes the initial EMA
            ema_acc = compute_ema(acc, ema_acc, alpha) 
            # Switch x and y axis for acceleration data
            acc_x, acc_y, acc_z = ema_acc
            ema_acc = np.array([acc_y, acc_x, acc_z])
            ema_acc = ema_acc / 9.8

            timestamp = time.time()
            curr_timestamp = sensor_data.update(device_id, [ema_acc], [ori], [timestamp, timestamp])
            sensor_data.update_virtual(device_id, ema_acc, ori) # update virtual data for display

            time_diff = curr_timestamp - prev_timestamp
            if time_diff >= min_time_diff:
                # send data via socket to live demo
                send_and_save_data(send_sock, sensor_data.virtual_acc, sensor_data.virtual_ori)
                prev_timestamp = curr_timestamp

        # Update and render
        manager.update(ema_acc)

        # Limit the frame rate
        clock.tick(60)

    # Cleanup
    teensy.serial_stop()
    pygame.quit()