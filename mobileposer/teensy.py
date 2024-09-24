"""Read IMU data from Teensy board."""

import serial
import time
import datetime
import math
import numpy as np
from collections import deque
import socket
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

# General Constants for converting back to floats from int16_t
INT16_T_MAX = 32767.0
INT16_T_SIZE = 2
EIT_MAX_VAL = 5.0
EIT_MIN_VAL = 0.0
QUAT_MAX_VAL = 1.0
QUAT_MIN_VAL = -1.0
ACC_MAX_VAL = 16.0
ACC_MIN_VAL = -16.0
GYR_MAX_VAL = 2000.0
GYR_MIN_VAL = -2000.0
MAG_MAX_VAL = 100.0
MAG_MIN_VAL = -100.0

ZERO_INDICES = [0,  1,  7,  8,  9, 10, 17, 18, 19, 26, 27, 28, 35, 36, 37, 44, 45, 46, 53, 54, 55, 56, 62, 63]
ZERO_INDICES_ERROR = [ 64,  65,  71,  72,  73,  74,  81,  82,  83,  90,  91,  92,  99, 100, 101, 108, 109, 110, 117, 118, 119, 120, 126, 127]

# TOTAL_DATA_POINTS = 64 + 64 + 13 # if eit error is included
# TOTAL_DATA_POINTS = 64 + 13
TOTAL_DATA_POINTS = 13
# INCOMING_BYTES = 186 # if eit error is included
INCOMING_BYTES = 106
# data = np.zeros(TOTAL_DATA_POINTS)

BLE_MODE = False


class TeensySerialManager:
    PORT_NAME = '/dev/cu.usbmodem131355501'
    DATA_DICT_ENTRY = 'bluetooth_data'
    DATA_DICT_ENTRY_IMU = 'imu_data'
    DATA_DICT_ENTRY_EIT = 'eit_data'

    def __init__(self, data_dict, numElect, serialPortName=PORT_NAME, data_dict_entry=DATA_DICT_ENTRY,
                 data_dict_entry_imu=DATA_DICT_ENTRY_IMU, data_dict_entry_eit=DATA_DICT_ENTRY_EIT) -> None:
        self.numElect = numElect
        self.bufferSize = self.numElect**2 + 13
        self.bufferSize_eit = self.numElect**2
        self.bufferSize_imu = 13
        self.data_dict = data_dict
        self.data_dict_entry = data_dict_entry 
        self.data_dict_entry_imu = data_dict_entry_imu
        self.data_dict_entry_eit = data_dict_entry_eit
        self.serialPortName = serialPortName

        self.recent_data = deque(maxlen=200)
        self.recent_data_imu = deque(maxlen=200)
        self.recent_data_eit = deque(maxlen=200)

        self.collect_data = False
        self.continue_running = True

        self.imu = np.zeros(13)

        self.timestamp_prev = 0
        self.fps = 0
        self.fps_total = np.zeros(100)
        self.i = 0

        self.timestamp_prev_imu = 0
        self.fps_imu = 0
        self.fps_total_imu = np.zeros(100)
        self.i_imu = 0

        self.timestamp_prev_eit = 0
        self.fps_eit = 0
        self.fps_total_eit = np.zeros(100)
        self.i_eit = 0

        # send data to UDP socket
        self.UDP_DATA_HOST = "127.0.0.1"
        self.UDP_DATA_PORT = 5005
        self.UDP_DATA_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP


    def serial_start(self):
        ser = None
        while ser is None:
            try:
                self.ser = serial.Serial(self.serialPortName, baudrate=230400)
                time.sleep(2)
                print(f'> EIT Serial Connection Established ... ({self.serialPortName})')
                return ser
            except serial.serialutil.SerialException as se:
                print("Serial Exception 0: ", str(se))
                time.sleep(1)
                continue

    def serial_read_val(self):
        try:
            res = self.ser.readline().decode().rstrip().split(',')
            if BLE_MODE:
                if '\x05' in res[0]:
                    res[0] = res[0].replace('\05','')
            res = [float(val) for val in res]
            
            if len(res) == self.bufferSize:
                linux_time = time.time()
                timestamp = datetime.datetime.now()
                self.process_data(res, linux_time)
                if self.timestamp_prev == 0:
                    self.timestamp_prev = timestamp
                    return
                else:
                    # if self.fps == 0:
                    fps = 1/((timestamp - self.timestamp_prev).total_seconds())
                    self.timestamp_prev = timestamp
                    self.i += 1
                    self.fps_total[int(self.i%100)] = fps
                    self.fps = sum(self.fps_total[self.fps_total != 0]) / sum(self.fps_total != 0)
                    # print(self.fps)
                # return res
            elif len(res) == self.bufferSize_eit:
                linux_time = time.time()
                timestamp = datetime.datetime.now()
                self.process_data(res, linux_time)
                if self.timestamp_prev_eit == 0:
                    self.timestamp_prev_eit = timestamp
                    return
                else:
                    # if self.fps == 0:
                    fps = 1/((timestamp - self.timestamp_prev_eit).total_seconds())
                    self.timestamp_prev_eit = timestamp
                    self.i_eit += 1
                    self.fps_total_eit[int(self.i_eit%100)] = fps
                    self.fps_eit = sum(self.fps_total_eit[self.fps_total_eit != 0]) / sum(self.fps_total_eit != 0)
                    # print(self.fps)
                    # print("EIT FPS: ", fps)
                # return res

                # self.process_data(res[0:self.bufferSize_imu], linux_time) # TODO: remove this when including IMU data
            elif len(res) == self.bufferSize_imu:
                linux_time = time.time()
                timestamp = datetime.datetime.now()
                self.process_data(res, linux_time)
                if self.timestamp_prev_imu == 0:
                    self.timestamp_prev_imu = timestamp
                    return
                else:
                    # if self.fps == 0:
                    fps = 1/((timestamp - self.timestamp_prev_imu).total_seconds())
                    self.timestamp_prev_imu = timestamp
                    self.i_imu += 1
                    self.fps_total_imu[int(self.i_imu%100)] = fps
                    # print("IMU FPS: ", fps)
                    self.fps_imu = sum(self.fps_total_imu[self.fps_total_imu != 0]) / sum(self.fps_total_imu != 0)
                    # print(self.fps)
                # return res
            else:
                print("Invalid buffer length: ", res)
                return
        except serial.serialutil.SerialException as se:
            print("Serial Exception 1: ", str(se))
            self.ser = self.serial_start()
            return
        except Exception as e:
            print("Invalid Data: ", str(e))
            return

    def dequantize_data(self, data_array, scalar, min, max):
        return [val*(max-min)/scalar + min for val in data_array]

    def process_data(self, combined_data, linux_time):
        data = combined_data
        # data = data + [0]*self.bufferSize_imu
        self.imu = data[self.bufferSize_eit:]

        if len(data) == self.bufferSize:
            data_dict_entry = self.data_dict_entry
            self.recent_data.append((linux_time, data))
            self.recent_data_eit.append((linux_time, data[:self.bufferSize_eit]))
            self.recent_data_imu.append((linux_time, data[self.bufferSize_eit:]))
        elif len(data) == self.bufferSize_eit:
            data_dict_entry = self.data_dict_entry_eit
            self.recent_data_eit.append((linux_time, data))
        elif len(data) == self.bufferSize_imu:
            data_dict_entry = self.data_dict_entry_imu
            self.recent_data_imu.append((linux_time, data))
        else:
            return
        
        if self.collect_data:
            if data_dict_entry not in self.data_dict.keys():
                if data_dict_entry == self.data_dict_entry:
                    self.data_dict[data_dict_entry] = [(linux_time, data)]
                    self.data_dict[self.data_dict_entry_eit] = [(linux_time, data[:self.bufferSize_eit])]
                    self.data_dict[self.data_dict_entry_imu] = [(linux_time, data[self.bufferSize_eit:])]
                else:
                    self.data_dict[data_dict_entry] = [(linux_time, data)]
            else:
                if data_dict_entry == self.data_dict_entry:
                    self.data_dict[data_dict_entry].append((linux_time, data))
                    self.data_dict[self.data_dict_entry_eit].append((linux_time, data[:self.bufferSize_eit]))
                    self.data_dict[self.data_dict_entry_imu].append((linux_time, data[self.bufferSize_eit:]))
                else:
                    self.data_dict[data_dict_entry].append((linux_time, data)) # np.array(combined_data), 0-63 is eit, 64-67 is quat, 68-70 is acc, 71-73 is gyr, 74-76 is mag


    def get_recent_data(self):
        list_recent_data = list(self.recent_data)
        list_recent_data.reverse()  # Reverse the list so that the most recent data is first
        return list_recent_data
    
    def get_recent_data_imu(self):
        list_recent_data = list(self.recent_data_imu)
        list_recent_data.reverse()
        return list_recent_data
    
    def get_recent_data_eit(self):
        list_recent_data = list(self.recent_data_eit)
        list_recent_data.reverse()
        return list_recent_data
    
    def get_fps(self): 
        return self.fps

    def get_fps_imu(self):
        return self.fps_imu
    
    def get_fps_eit(self):
        return self.fps_eit

    def start_data_collection(self):
        self.collect_data = True

    def stop_data_collection(self):
        self.collect_data = False

    def get_connected_status(self):
        return True

    def run(self):
        self.ser = self.serial_start()
        while True:
            res = self.serial_read_val()
            time.sleep(0.005)
            try:
                dat = str(self.get_recent_data_eit()[0][1])
                imu = self.get_recent_data_imu()
                print("IMU: ")
                print(type(imu))
                eitBytes = bytes(dat, encoding="utf8")
                self.UDP_DATA_socket.sendto(eitBytes, (self.UDP_DATA_HOST, self.UDP_DATA_PORT))
            except:
                pass