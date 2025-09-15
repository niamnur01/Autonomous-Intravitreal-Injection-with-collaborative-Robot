#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class EyeGazeRecorder(Node):

    def __init__(self):
        super().__init__('eye_gaze_recorder')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/eye_gaze',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.data = []
        self.start_time = time.time()
        self.duration = 45  # seconds

        self.get_logger().info('Recording eye gaze data for 45 seconds...')

    def listener_callback(self, msg):
        current_time = time.time()
        if current_time - self.start_time <= self.duration:
            self.data.append(msg.data)
        else:
            self.save_data()
            self.get_logger().info('Finished recording. Data saved to eye_movement.txt')
            rclpy.shutdown()

    def save_data(self):
        with open("eye_movement.txt", "w") as f:
            for gaze in self.data:
                f.write(f"{gaze[0]}, {gaze[1]}\n")

def main(args=None):
    rclpy.init(args=args)
    recorder = EyeGazeRecorder()
    rclpy.spin(recorder)
    recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
