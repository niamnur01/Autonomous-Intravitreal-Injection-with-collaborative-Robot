#!/usr/bin/env python3
"""
eye_gaze_replayer.py
Publishes prerecorded gaze data on /eye_gaze **and** serves the latest
orientation quaternion on /gazeSrv (Pose service).
"""

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayLayout

# --- NEW imports for the service -------------------------------------------
from project_interfaces.srv import Pose            # custom Pose service
from transforms3d import euler                     # euler2quat
# ---------------------------------------------------------------------------


class EyeGazeReplayer(Node):

    def __init__(self):
        super().__init__('eye_gaze_replayer')

        # ────── publisher (unchanged) ───────────────────────────────────────
        self.publisher_ = self.create_publisher(Float64MultiArray,
                                                '/eye_gaze', 10)
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "test", "eye_movement.txt")
        self.data = self.load_data(file_path)
        
        self.timer_period = 0.1    # 10 Hz
        self.index = 0

        if not self.data:
            self.get_logger().error("No data found in eye_movement.txt")
            rclpy.shutdown()
            return

        # ────── state variables that the service will read ──────────────────
        self.roll = 0.0            # fixed at zero
        self.pitch = 0.0
        self.yaw = 0.0

        # ────── service server NEW ──────────────────────────────────────────
        self.srv = self.create_service(Pose,          # service type
                                       'gazeSrv',     # service name
                                       self.service_callback)

        # ────── timer to replay data (unchanged apart from state update) ───
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info('Replaying eye gaze data in a loop…')

    # ════════════════════════════════════════════════════════════════════════
    # I/O helpers
    # ════════════════════════════════════════════════════════════════════════
    def load_data(self, filename):
        gaze_data = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        try:
                            x = float(parts[0].strip())  # yaw
                            y = float(parts[1].strip())  # pitch
                            gaze_data.append([x, y])
                        except ValueError:
                            continue        # skip malformed lines
        except FileNotFoundError:
            self.get_logger().error(f"File '{filename}' not found.")
        return gaze_data

    # ════════════════════════════════════════════════════════════════════════
    # Main loop: publish gaze sample at 10 Hz
    # ════════════════════════════════════════════════════════════════════════
    def timer_callback(self):
        """Publish one sample and remember it for the service."""
        self.yaw, self.pitch = self.data[self.index]  # update state

        msg = Float64MultiArray()
        msg.data = [self.yaw, self.pitch]             # [yaw, pitch]
        self.publisher_.publish(msg)

        self.index = (self.index + 1) % len(self.data)

    # ════════════════════════════════════════════════════════════════════════
    # Service utilities
    # ════════════════════════════════════════════════════════════════════════
    def get_quaternion(self):
        """Convert current RPY (roll = 0) to quaternion (w, x, y, z)."""
        w, x, y, z = euler.euler2quat(self.roll,
                                      -self.pitch,
                                      self.yaw,
                                      'rzyx')
        return w, x, y, z

    def service_callback(self, request, response):
        """Return the latest orientation as a quaternion."""
        response.w, response.x, response.y, response.z = self.get_quaternion()
        self.get_logger().info(
            'Orientation sent: [w: {:.3f}, x: {:.3f}, y: {:.3f}, z: {:.3f}]'.format(
                response.w, response.x, response.y, response.z))
        return response


# ═══════════════════════════════════════════════════════════════════════════
# Boilerplate
# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    replayer = EyeGazeReplayer()
    rclpy.spin(replayer)
    replayer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
