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

min_time_diff = 0.1  # Example: 100ms

num_devices = len(DEVICE_POSITIONS)
per_device_counters = {device_id: 0 for device_id in range(num_devices)}
per_device_delay_sums = {device_id: 0.0 for device_id in range(num_devices)}
last_log_time = time.time()

# NEW: Import queue for threading
from queue import Queue

# NEW: Create a thread-safe queue for incoming data
data_queue = Queue()

# NEW: Function to receive data from sockets and put it into the queue
def data_receiver(sockets):
    empty = []
    while True:
        try:
            readable, _, _ = select.select(sockets, empty, empty, 0.001)
            for sock in readable:
                while True:
                    try:
                        data, addr = sock.recvfrom(CHUNK, socket.MSG_DONTWAIT)
                        receive_time = time.time()
                        data_queue.put((data, addr, receive_time))
                    except BlockingIOError:
                        break  # No more data to read from this socket
        except Exception as e:
            print("Data receiver encountered an exception:", e)

# NEW: Start the data receiver thread
def start_data_receiver_thread(sockets):
    receiver_thread = threading.Thread(target=data_receiver, args=(sockets,), daemon=True)
    receiver_thread.start()

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    # Init socket and data handlers
    sockets = init_sockets(HOST, PORTS)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sensor_data = SensorData()

    # NEW: Start the data receiver thread
    start_data_receiver_thread(sockets)

    # Setup PyGame manager
    manager = PyGameManager(860, 860)

    # Initialize cubes for each device
    cubes = [Cube(device_id=i, position=DEVICE_POSITIONS[i]) for i in range(len(DEVICE_POSITIONS))]
    for cube in cubes:
        manager.add_cube(cube)

    frames = 0
    prev_timestamp = 0
    curr_timestamp = 0
    # Initialize drawing time accumulator
    total_draw_time = 0.0

    while True:
        continue_running, key_c_pressed = manager.handle_events()
        if not continue_running:
            break

        if key_c_pressed:
            print("Started Calibration!")
            sensor_data.calibrate()

        # NEW: Measure how many messages are in the queue
        queue_size = data_queue.qsize()

        # NEW: Process all available data in the queue
        while not data_queue.empty():
            try:
                data, addr, receive_time = data_queue.get_nowait()

                # NEW: Start timing the processing
                process_start_time = time.time()

                # Process received data
                vis_str, device_id, curr_acc, curr_ori, timestamps = process_data(data)
                curr_timestamp = sensor_data.update(device_id, curr_acc, curr_ori, timestamps)
                glb_ori, glb_acc = sensor2global(
                    sensor_data.get_orientation(device_id),
                    sensor_data.get_acceleration(device_id),
                    sensor_data.calibration_quats,
                    device_id
                )
                sensor_data.update_virtual(device_id, glb_acc, glb_ori)  # Update virtual data for display

                # Update cube orientation
                cubes[device_id].set_orientation(sensor_data.virtual_ori[device_id])

                # Calculate delay
                unix_timestamp = timestamps[0]
                delay = receive_time - unix_timestamp  # Delay in seconds

                # Update frequency and delay tracking
                per_device_counters[device_id] += 1
                per_device_delay_sums[device_id] += delay

                time_diff = curr_timestamp - prev_timestamp

                if time_diff >= min_time_diff:
                    # Send data via socket to live demo
                    send_and_save_data(send_sock, sensor_data.virtual_acc, sensor_data.virtual_ori)
                    prev_timestamp = curr_timestamp

                # NEW: End timing the processing
                process_end_time = time.time()
                process_duration = process_end_time - process_start_time

                # NEW: Accumulate total processing time (optional)
                total_processing_time = total_processing_time + process_duration if 'total_processing_time' in locals() else process_duration

                data_queue.task_done()

            except Exception as e:
                print("Exception encountered during data processing: ", e)

        draw_start_time = time.time()

        # Draw cubes using the PyGameManager
        manager.update(glb_acc)

        draw_end_time = time.time()
        draw_duration = draw_end_time - draw_start_time
        total_draw_time += draw_duration

        # Frequency and Delay Logging
        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            for device_id in range(num_devices):
                count = per_device_counters[device_id]
                total_delay = per_device_delay_sums[device_id]
                if count == 0 and total_delay == 0:
                    continue

                frequency = count / (current_time - last_log_time) if (current_time - last_log_time) > 0 else 0
                average_delay = (total_delay / count) if count > 0 else 0

                print(f"Device ID {device_id}: Frequency = {frequency:.2f} Hz, "
                      f"Average Delay = {average_delay*1000:.2f} ms")
            # NEW: Print total processing time
            if 'total_processing_time' in locals():
                total_processing_time = 0.0  # Reset processing time accumulator

            # Reset counters and delay sums
            per_device_counters = {device_id: 0 for device_id in range(num_devices)}
            per_device_delay_sums = {device_id: 0.0 for device_id in range(num_devices)}
            total_draw_time = 0.0  # Reset drawing time accumulator

            last_log_time = current_time

