
# Author: Salih Marangoz
# Date: 24 Dec 2020
# 
# Use W/A/S/D to drive the robot
# Yellow circle represents the real robot position and heading
# Blue circle represents noisy (odom) robot position
# Red circles represents landmarks
# Blue lines represents noisy measurements plotted according to the real robot position
#
# Close the window to end the simulation

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen
import sys
import numpy as np


def main():
    sim_params = {'sensor_max_range': 3.5,                          # in meters
                  'friction': np.array([0.4, 1.0, 0.3]),            # x,y,theta (y is not used since the robot model is not omni)
                  'dt': 1.0/10.0,                                   # simulator time step. Hint: 1.0/{FPS}
                  'lin_acc': 0.75,                                  # Linear acceleration
                  'ang_acc': 0.75,                                  # Angular acceleration
                  'odom_noise': np.array([0.005, 0.001, 0.005]),    # Odometry noise for [rotation_1, translation, rotation_2]
                  'sensor_noise': np.array([0.01, 0.01]),           # Sensor noise for [range, bearing]
                  'sensor_coverage': np.array([-np.pi*0.666, np.pi*0.666])      # Sensor coverage in [min_angle, max_angle]
                  }

    gui_params = {'width': 800,
                  'height': 600,
                  'window_resolution': 0.02,                        # meters per pixel
                  'bias_x': 100,                                    # move the map on x axis
                  'bias_y': -100                                    # move the map on y axis
                  }

    world_filename = 'world.dat'
    sensor_filename = 'sensor_data.dat'

    app = QApplication(sys.argv)
    sim = Simulator(world_filename, sensor_filename, sim_params, gui_params)
    sys.exit(app.exec())



class Simulator(QMainWindow):
    def __init__(self, world_filename, sensor_filename, sim_params, gui_params):
        super().__init__()

        # INIT FILE I/O
        self.world_filename = world_filename
        self.sensor_filename = sensor_filename
        self.sensor_file = open(self.sensor_filename, "w")

        # INIT SIMULATOR
        for k,v in sim_params.items():                          # set dictionary keys as object variable and dictionary values as variable values
            setattr(self, k, v)

        self.r_odom = np.array([0.0, 0.0, 0.0])                 # start value of noisy position   : [x, y, theta]
        self.r_pos = np.array([0.0, 0.0, 0.0])                  # start value of real position    : [x, y, theta]
        self.r_vel = np.array([0.0, 0.0, 0.0])                  # start value of real velocity    : [x, y, theta]
        self.r_acc = np.array([0.0, 0.0, 0.0])                  # start value of real acceleration: [x, y, theta]
        self.landmarks = self.readWorld(self.world_filename)    # landmarks read from file in [x,y]
        self.detected_landmarks = {}                            # currently detected landmarks in [x,y]
        self.noisy_landmarks = {}                               # currently detected and noise added landmarks in [x,y]

        # INIT GUI
        for k,v in gui_params.items():                          # set dictionary keys as object variable and dictionary values as variable values
            setattr(self, k, v)

        self.pressed_keys = {}                                  # Currently pressed keys in dictionary format where key values are the Qt key values
        self.title = "Simulator"
        self.InitWindow()
        self.timer = QTimer()
        self.timer.timeout.connect(self.timeoutEvent)
        self.timer.start(self.dt*1000.0)

    ##################### SIMULATOR ########################

    # Normalize the angle between [-2*pi,2*pi]
    def normalizeAngle(self, angle):
        while angle > np.pi:
            angle = angle - 2 * np.pi
        while angle < -np.pi:
            angle = angle + 2 * np.pi
        return angle


    # Move robot using acceleration command, update real and noisy robot positions, return noisy odometry commands in [rot1, trans, rot2]
    def moveRobot(self, acc_cmd):
        # Calculate robot movement
        d_x = (self.r_vel[0]*self.dt + 0.5*acc_cmd[0]*self.dt**2) * np.cos(self.r_pos[2])
        d_y = (self.r_vel[0]*self.dt + 0.5*acc_cmd[0]*self.dt**2) * np.sin(self.r_pos[2])
        d_theta = self.r_vel[2]*self.dt + 0.5*acc_cmd[2]*self.dt # update angle
        self.r_vel *= self.friction**self.dt
        self.r_vel += acc_cmd*self.dt
        
        # Calculate real position
        new_r_pos = self.r_pos + np.array([d_x, d_y, d_theta])
        new_r_pos[2] = self.normalizeAngle(new_r_pos[2])

        # Calculate real odom
        real_rot1  = np.arctan2(new_r_pos[1] - self.r_pos[1], new_r_pos[0] - self.r_pos[0]) - self.r_pos[2]
        real_trans = np.linalg.norm(new_r_pos[:2] - self.r_pos[:2]) 
        real_rot2  = new_r_pos[2] - self.r_pos[2] - real_rot1

        # Add noise
        noisy_rot1  = real_rot1  + np.random.normal(0.0, self.odom_noise[0])
        noisy_trans = real_trans + np.random.normal(0.0, self.odom_noise[1])
        noisy_rot2  = real_rot2  + np.random.normal(0.0, self.odom_noise[2])

        # Calculate noisy odom
        new_r_odom = [0.0, 0.0, 0.0]
        new_r_odom[0] = self.r_odom[0] + noisy_trans * np.cos(self.r_odom[2]+noisy_rot1)
        new_r_odom[1] = self.r_odom[1] + noisy_trans * np.sin(self.r_odom[2]+noisy_rot1)
        new_r_odom[2] = self.normalizeAngle( self.r_odom[2] + noisy_rot1 + noisy_rot2 )

        # Update values
        self.r_pos = new_r_pos
        self.r_odom = new_r_odom

        odom_cmd = np.array([noisy_rot1, noisy_trans, noisy_rot2])
        return odom_cmd


    # Calculate noisy measurements according to the current robot pose and real landmark positions, also return detected real landmarks without noise
    def getMeasurements(self):
        # Find detected landmarks according to the distance
        detected_landmarks = {}
        for idx, l in self.landmarks.items():
            dist = np.linalg.norm(self.r_pos[:2] - l)
            if (dist < self.sensor_max_range):
                detected_landmarks[idx] = l

        # Calculate measurement values
        measurements = {}
        for idx, l in detected_landmarks.items():
            z_range = np.linalg.norm(l - self.r_pos[:2])
            angle = np.arctan2(l[1] - self.r_pos[1], l[0] - self.r_pos[0]) - self.r_pos[2]

            # Add gaussian noise
            z_range += np.random.normal(0.0, self.sensor_noise[0])
            angle += np.random.normal(0.0, self.sensor_noise[1])

            z_bearing = self.normalizeAngle(angle)
            measurements[idx] = [z_range, z_bearing]

        # Filter measurements and detected landmarks according to the sensor coverage limits
        to_be_deleted = []
        for idx, l in measurements.items():
            if (l[1] < self.sensor_coverage[0] or l[1] > self.sensor_coverage[1]):
                to_be_deleted.append(idx)
        for idx in to_be_deleted:
            del measurements[idx]
            del detected_landmarks[idx]

        return detected_landmarks, measurements


    # Calculate landmark positions according the measurement range and bearins and robot position
    def calculateLandmarks(self, measurements, robot_pos):
        landmarks = {}
        for k,v in measurements.items():
            x = v[0] * np.cos(v[1] + robot_pos[2]) + robot_pos[0]
            y = v[0] * np.sin(v[1] + robot_pos[2]) + robot_pos[1]
            landmarks[k] = [x,y]
        return landmarks

    # This function runs every dt (time resolution) to simulate the robot
    def simulationTick(self, acc_cmd):
        odom_cmd = self.moveRobot(acc_cmd)
        self.detected_landmarks, measurements = self.getMeasurements()
        self.noisy_landmarks = self.calculateLandmarks(measurements, self.r_pos)
        self.saveSensor(odom_cmd, measurements)


    ######################## GUI ###########################

    # Converts metric positions to pixel positions on the window
    def m2px(self, x,y):
        height = self.frameGeometry().height()
        return (x / self.window_resolution)+self.bias_x, (height - y / self.window_resolution)+self.bias_y


    # Init Qt window
    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height)
        self.show()


    # Paints the given landmark to the window
    def paintLandmark(self, painter, x, y, landmark_size=0.25):
        x_px, y_px = self.m2px(x, y)
        landmark_size_px = landmark_size / self.window_resolution

        painter.setPen(QPen(Qt.black,  3, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.drawEllipse(x_px-landmark_size_px/2, y_px-landmark_size_px/2, landmark_size_px, landmark_size_px)


    # Paints the given robot to the window
    def paintRobot(self, painter, x, y, theta, robot_size=0.5, robot_color=Qt.yellow):
        robot_heading_size = robot_size*0.75
        x_end = x + robot_heading_size*np.cos(theta)
        y_end = y + robot_heading_size*np.sin(theta)

        robot_size_px = robot_size / self.window_resolution
        robot_heading_size_px = robot_heading_size / self.window_resolution
        x_px, y_px = self.m2px(x, y)
        x_end_px, y_end_px = self.m2px(x_end, y_end)

        # paint body
        painter.setPen(QPen(Qt.black,  3, Qt.SolidLine))
        painter.setBrush(QBrush(robot_color, Qt.SolidPattern))
        painter.drawEllipse(x_px-robot_size_px/2, y_px-robot_size_px/2, robot_size_px, robot_size_px)

        # paint robot heading
        painter.setPen(QPen(Qt.black,  3, Qt.SolidLine))
        painter.drawLine(x_px, y_px, x_end_px, y_end_px)


    # Paints the given sensor measurement as a line between robot and landmark positions
    def paintSensor(self, painter, r_x, r_y, l_x, l_y):
        r_x_px, r_y_px = self.m2px(r_x, r_y)
        l_x_px, l_y_px = self.m2px(l_x, l_y)

        painter.setPen(QPen(Qt.blue,  1, Qt.SolidLine))
        painter.drawLine(r_x_px, r_y_px, l_x_px, l_y_px)


    ##################### GUI EVENTS #######################

    def keyPressEvent(self, e):
        key = e.key()
        self.pressed_keys[key] = True


    def keyReleaseEvent(self, e):
        key = e.key()
        del self.pressed_keys[key]


    # Timer callback function to get user input, tick the simulation and refresh the drawings
    def timeoutEvent(self):
        acc_cmd = np.array([0.0, 0.0, 0.0])

        for k in self.pressed_keys.keys():
            if k == Qt.Key_W:
                acc_cmd[0] = self.lin_acc
            if k == Qt.Key_S:
                acc_cmd[0] = -self.lin_acc
            if k == Qt.Key_A:
                acc_cmd[2] = self.ang_acc
            if k == Qt.Key_D:
                acc_cmd[2] = -self.ang_acc

        self.simulationTick(acc_cmd)
        self.update() # redraw the widget


    # This event is called when repainting is needed
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        #for idx, l in self.detected_landmarks.items():
        #    self.paintSensor(painter, self.r_pos[0], self.r_pos[1], l[0], l[1])
        for idx, l in self.noisy_landmarks.items():
            self.paintSensor(painter, self.r_pos[0], self.r_pos[1], l[0], l[1])

        for idx, l in self.landmarks.items():
            self.paintLandmark(painter, l[0], l[1])

        self.paintRobot(painter, self.r_odom[0], self.r_odom[1], self.r_odom[2], robot_color=Qt.blue) # paint noisy pose
        self.paintRobot(painter, self.r_pos[0], self.r_pos[1], self.r_pos[2])


    ##################### FILE I/O #########################

    # Reads world file. Every line is in this format:
    # [LANDMARK_ID] [LANDMARK_X] [LANDMARK_Y]
    def readWorld(self, filename):
        landmarks = {}
        with open(filename) as file:
            for line in file:
                line_s = line.split()
                landmarks[int(line_s[0])] = [float(line_s[1]), float(line_s[2])]
        return landmarks

    # Saves odom and measurements. One timestep includes a single odom and single/multiple measurements. One timestep is in this format:
    # ODOMETRY [ROT_1] [TRANS] [ROT_2]
    # SENSOR [LANDMARK_ID] [RANGE] [BEARING]
    def saveSensor(self, u, z):
        self.sensor_file.write("ODOMETRY "+ str(u[0]) +" "+ str(u[1]) +" "+ str(u[2]) +"\n")
        for k,v in z.items():
            self.sensor_file.write("SENSOR "+ str(k) +" "+ str(v[0]) +" "+ str(v[1]) +"\n")



if __name__ == "__main__":
    main()