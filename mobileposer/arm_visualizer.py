import sys
import socket
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl

class ArmVisualizer:
    def __init__(self):
        self.app = QApplication([])
        self.view = gl.GLViewWidget()
        self.view.show()
        self.view.setWindowTitle('Real-Time Arm Joints Visualization')

        # Set top-down camera position and view
        self.camera_distance = 5
        # self.view.setCameraPosition(distance=1, elevation=90, azimuth=0)  # Top-down view

        # Create scatter plot for joints
        self.joints_plot = gl.GLScatterPlotItem(size=3, color=(1, 0, 0, 1), pxMode=False)
        self.view.addItem(self.joints_plot)

        # Create line plot for bones
        self.bones_plot = gl.GLLinePlotItem(color=(1, 1, 1, 1), width=10)
        self.view.addItem(self.bones_plot)

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update every 50ms for smoother updates

        # Placeholder for rarm_joints data
        self.rarm_joints = np.zeros((5, 3))  # Initialize to zeros, replace with actual data

    def update_plot(self):
        cam_pos = self.view.cameraPosition()
        self.joints_plot.setData(pos=self.rarm_joints)  # Update joints
        self.bones_plot.setData(pos=self.rarm_joints)   # Update bones

    def update_joints(self, rarm_joints):
        """ Update joint data with the received arm joints. """
        self.rarm_joints = rarm_joints

    def start(self):
        sys.exit(self.app.exec_())

class JointReceiver(QtCore.QThread):
    new_joints = QtCore.pyqtSignal(np.ndarray)  # Signal to emit new joint data

    def __init__(self, server_address='127.0.0.1', server_port=5005):
        super().__init__()
        self.server_address = server_address
        self.server_port = server_port

    def run(self):
        """The method that runs in the thread and listens for incoming joint data."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.server_address, self.server_port))

        while True:
            data, addr = sock.recvfrom(1024)
            rarm_joints = pickle.loads(data)
            self.new_joints.emit(rarm_joints)  # Emit the received joints through a signal


if __name__ == '__main__':
    # Initialize the visualizer
    visualizer = ArmVisualizer()

    # Initialize the joint receiver
    joint_receiver = JointReceiver(server_address='127.0.0.1', server_port=5005)

    # Connect the joint receiver signal to the visualizer's update_joints method
    joint_receiver.new_joints.connect(visualizer.update_joints)

    # Start the joint receiver thread
    joint_receiver.start()

    # Start the PyQt5 visualizer
    visualizer.start()
