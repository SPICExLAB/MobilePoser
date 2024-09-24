import os


# Calibration constants
KEYS = ['unix_timestamp', 'sensor_timestamp', 'accel_x', 'accel_y', 'accel_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', "roll", "pitch", "yaw"]
STOP = "stop"
SEP = ":"

# Socket configurations
OUT_UDP_PORT = 7777
OUT_UDP_IP = "127.0.0.1"
HOST = "0.0.0.0"
PORTS = [8001, 8002, 8003, 8004, 8005]

# Buffer settings
CHUNK = 2048
BUFFER_SIZE = 50
min_time_diff = 1/25.6  # seconds

# Display settings
H, W = 860, 860