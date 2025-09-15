#!/usr/bin/env python3

import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render, getDataset

import rclpy
from rclpy.node import Node

import time
import threading
from collections import deque
from transforms3d import euler

from project_interfaces.srv import Pose
from std_msgs.msg import Float64MultiArray, MultiArrayLayout

import pyquaternion as pyq

from filterpy.kalman import KalmanFilter

ROLL = 0.0
PITCH = 0.0
YAW = 0.0

ROUTINE_FREQUENCY = 20

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default='cuda', type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def eye_tracking():
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id

    gaze_pipeline = Pipeline(
        weights = getDataset(),
        arch='ResNet50',
        device = torch.device(args.device)
    )

    # Checking camera index: 'ls -al /dev/video*'
    # Adding '--cam N_CAMERA' to command line to change camera
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    global ROLL 
    global PITCH
    global YAW

    with torch.no_grad():
        while True:
            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results = gaze_pipeline.step(frame)
            ROLL = 0.0
            PITCH = results.pitch[0]
            YAW = results.yaw[0]

            # Visualize output
            frame = render(frame, results)
           
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  

class MinimalService(Node):

    def __init__(self):
        # Initializing class
        super().__init__('minimal_service')

        # Creating buffer for smoothing the orientation
        self.buffer = deque()
        self.buffer.append(np.array([ROLL,PITCH,YAW]))

        # Creating service to deliver eye orientation
        self.srv = self.create_service(Pose, 'gazeSrv', self.service_callback)

        # Creating publisher to publish Yaw and Pitch to /eye_gaze topic
        self.gaze_publisher = self.create_publisher(Float64MultiArray, 'eye_gaze', 10)

        # Setting angle limit to 70°, which is nearly 1.22 in radiant
        self.angle_limit = 1.22

        # Creating timer to perform routine
        self.timer = self.create_timer(1/ROUTINE_FREQUENCY, self.routine)

        # Creating Kalman Filter 
        self.kalman_filter =  self.create_filter()

    def routine(self):
        self.roll = self.angle_limit_control(ROLL, self.angle_limit)
        self.pitch = self.angle_limit_control(PITCH, self.angle_limit)
        self.yaw = self.angle_limit_control(YAW, self.angle_limit)

        z = np.array([self.roll, self.pitch, self.yaw])
        self.kalman_filter.predict()
        self.kalman_filter.update(z)

        x = self.kalman_filter.x
        self.roll, self.pitch, self.yaw = x[0], x[1], x[2]

        self.publish_eye_gaze(self.yaw, self.pitch)

    def angle_limit_control(self, angle, limit_rad):
        # Controlling limit
        if angle > limit_rad:
            angle = limit_rad

        if angle < -limit_rad:
            angle = -limit_rad

        return angle
    
    def create_filter(self):
        filter = KalmanFilter (dim_x=6, dim_z=3)

        z = np.array([ROLL, PITCH, YAW])

        v = np.array([0.0, 0.0, 0.0])
        
        filter.x = np.concatenate((z, v), axis=0)

        filter.R = np.array([[5.0, 0.0, 0.0],
                             [0.0, 5.0, 0.0],
                             [0.0, 0.0, 5.0]])
        
        filter.P = np.array([[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1000.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 1000.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 1000.0]])

        filter.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        
        d_t = 0.1
        filter.F = np.array([[1.0, 0.0, 0.0, d_t, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, d_t, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0, d_t],
                             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        return filter

    def get_quaternion(self):
        w, x, y, z = euler.euler2quat(self.roll, self.pitch, self.yaw, 'rzyx')

        #print("[roll: %f, pitch: %f, yaw: %f]" % (self.roll, self.pitch, self.yaw))
        #print("[w: %f, x: %f, y: %f, z: %f]" % (w, x, y, z))    

        return w, x, y, z

    def service_callback(self, request, response):
        # Service required, responding with current orientation
        response.w, response.x, response.y, response.z = self.get_quaternion()
        self.get_logger().info('Orientation sended: [w: %f, x: %f, y: %f, z: %f]' % (response.w, response.x, response.y, response.z))

        return response
    
    def publish_eye_gaze(self, yaw, pitch):
        gaze_msg = Float64MultiArray()
        gaze_msg.layout = MultiArrayLayout(dim=[], data_offset=0)
        gaze_msg.data = [yaw, pitch]
        self.gaze_publisher.publish(gaze_msg)


def main(args=None):
    t1 = threading.Thread(target=eye_tracking,args=())
    t1.start()

    rclpy.init()
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)

    t1.join()

    rclpy.shutdown()

if __name__ == '__main__':
    main()