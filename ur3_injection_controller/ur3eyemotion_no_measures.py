#!/usr/bin/env python3
"""
ROS2 node that continuously queries the Pose service, prints the returned quaternion,
pre-rotates it, and visualizes the eye frame, eyeball, iris, and pupil in RViz.
"""
import rclpy
from rclpy.node import Node
from project_interfaces.srv import Pose
import time

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.transform_broadcaster import TransformBroadcaster

import numpy as np
import transforms3d.euler as euler
import pyquaternion as pyq
import transforms3d.quaternions as quat


# Constants
FREQUENCY = 20  # polling rate in Hz
TIME_STEP = 1.0 / FREQUENCY
EYE_FRAME = 'eye_frame'
PARENT_FRAME = 'base_link'
EYE_POSITION = (-0.164, 0.35274, 0.629)  # [m]
EYE_RADIUS = 0.015  # [m]
SIZE = 1.0  # scaling factor for markers

class PosePrinter(Node):
    def __init__(self):
        super().__init__('pose_printer')

        # for 5-second logging
        self._last_log_time = time.time()

        # Create client for the Pose service
        self.cli = self.create_client(Pose, 'gazeSrv')
        # Wait until the service is available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.get_logger().info('Connected to Pose service.')

        # Prepare request
        self.req = Pose.Request()

        ##Define vector rotation for injection trajectory, prepare rotation matrix for environment_building
        yaw_deg = 50# or any value within [-80, 80] for left eye, and [100, 260] for right eye
        pitch_deg = 45.5 # or a sample within [44.5, 46.5]
        R_pitch = euler.euler2mat(0, np.deg2rad(pitch_deg), 0, 'sxyz')
        R_yaw = euler.euler2mat(0, 0, np.deg2rad(yaw_deg), 'sxyz')
        self.R_inject = R_yaw @ R_pitch

        # RViz publishers
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.current_position = EYE_POSITION

    def quaternion_multiply(self, q1_arr, q0_arr):
        q1 = pyq.Quaternion(q1_arr)
        q0 = pyq.Quaternion(q0_arr)
        q = q1 * q0
        return np.array([q.w, q.x, q.y, q.z])

    def run(self):
        # Main loop
        while rclpy.ok():
            # Call service
            future = self.cli.call_async(self.req)
            rclpy.spin_until_future_complete(self, future)

            if future.done() and future.result() is not None:
                resp = future.result()
                # Raw quaternion
                eye_q = np.array([resp.w, resp.x, resp.y, resp.z])
                # Print
                #self.get_logger().info(
                #    f"Received quaternion -> w: {resp.w:.4f}, x: {resp.x:.4f}, y: {resp.y:.4f}, z: {resp.z:.4f}"
                #)
                # Visualize
                self.environment_building(eye_q)
            else:
                self.get_logger().warn('Service call failed or no response.')

            time.sleep(TIME_STEP)

    def publish_eye_and_arrow(self, eyeball, iris, pupil, arrow):
        ma = MarkerArray()
        ma.markers.extend([eyeball, iris, pupil, arrow])
        self.marker_array_pub.publish(ma)

    def environment_building(self, eye_q):
        # Compute pre-rotation quaternion based on eye position to set the spawning eye orientation
        q_trans = euler.euler2quat(np.pi, 0, -np.pi/2, 'rzyx')  # returns w,x,y,z
        # Combined quaternion for eye gaze and eye frame rotation (for injection_vector)
        w, x, y, z = self.quaternion_multiply(q_trans, eye_q)

        #Create the vector to visualize the injection trajectory
        ##rotate the local [0, 0, EYE_RADIUS] into the world
        v_rotated = self.R_inject @ np.array([0, 0, 1])
        q_frame = [w, x, y, z]
        R_frame = quat.quat2mat(q_frame)
        v_world = R_frame @ v_rotated

        #injection point
        position_world = self.current_position + (v_world*EYE_RADIUS*3) #by multiplying *3 v_world, the "injection point" is outside the eye, to visualize the injection vector arrow

        # Publish transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = PARENT_FRAME
        t.child_frame_id = EYE_FRAME
        t.transform.translation.x = self.current_position[0]
        t.transform.translation.y = self.current_position[1]
        t.transform.translation.z = self.current_position[2]
        t.transform.rotation.w = w
        t.transform.rotation.x = x
        t.transform.rotation.y = y
        t.transform.rotation.z = z
        self.tf_broadcaster.sendTransform(t)

        # 1) Eyeball
        eyeball = Marker()
        eyeball.header.frame_id = EYE_FRAME
        eyeball.header.stamp    = t.header.stamp
        eyeball.ns   = 'eye_viz'
        eyeball.id   = 0
        eyeball.type = Marker.SPHERE
        eyeball.action            = Marker.ADD
        eyeball.pose.position.x   = 0.0
        eyeball.pose.position.y   = 0.0
        eyeball.pose.position.z   = 0.0
        eyeball.pose.orientation.w = 1.0
        eyeball.scale.x = 2 * EYE_RADIUS
        eyeball.scale.y = 2 * EYE_RADIUS
        eyeball.scale.z = 2 * EYE_RADIUS
        eyeball.color.a = 1.0
        eyeball.color.r = 1.0
        eyeball.color.g = 1.0
        eyeball.color.b = 1.0

        # 2) Iris
        iris = Marker()
        iris.header = eyeball.header
        iris.ns     = eyeball.ns
        iris.id     = 1
        iris.type   = Marker.SPHERE
        iris.action            = Marker.ADD
        iris.pose.position.x   = 0.0
        iris.pose.position.y   = 0.0
        iris.pose.position.z   = EYE_RADIUS
        iris.pose.orientation.w = 1.0
        iris.scale.x = 0.015 * SIZE
        iris.scale.y = 0.015 * SIZE
        iris.scale.z = 0.006  * SIZE
        iris.color.a = 1.0
        iris.color.r = 0.0
        iris.color.g = 150.0/255.0
        iris.color.b = 0.0

        # 3) Pupil
        pupil = Marker()
        pupil.header = eyeball.header
        pupil.ns     = eyeball.ns
        pupil.id     = 2
        pupil.type   = Marker.SPHERE
        pupil.action            = Marker.ADD
        pupil.pose.position.x   = 0.0
        pupil.pose.position.y   = 0.0
        pupil.pose.position.z   = EYE_RADIUS + 0.003
        pupil.pose.orientation.w = 1.0
        pupil.scale.x = 0.006   * SIZE
        pupil.scale.y = 0.006   * SIZE
        pupil.scale.z = 0.0015 * SIZE
        pupil.color.a = 1.0
        pupil.color.r = 0.0
        pupil.color.g = 0.0
        pupil.color.b = 0.0

        #Arrow for injection trajectory
        arrow = Marker()
        arrow.header.frame_id = PARENT_FRAME
        arrow.header.stamp = t.header.stamp
        arrow.ns = 'injection_vector'
        arrow.id = 3
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        #start at the eye center, end at the injection point
        start = Point(x=self.current_position[0], y=self.current_position[1], z=self.current_position[2])
        end = Point(x=position_world[0], y=position_world[1], z=position_world[2])
        arrow.points = [start, end]
        #shaft, head diameter and lenght
        arrow.scale.x = 0.002 #shaft
        arrow.scale.y = 0.005 #head
        arrow.scale.z = 0.01 #lenght
        arrow.color.r = 1.0
        arrow.color.g = 0.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0

        # build a rotation from [0,0,1]→v_world
        v_inj = np.array([start.x-end.x, start.y-end.y, start.z-end.z])
        v_norm = v_inj / np.linalg.norm(v_inj)
        '''self.get_logger().info(
            f"Normalized injetion vector"
            f"[{v_norm[0]:.4f}, {v_norm[1]:.4f}, {v_norm[2]:.4f}]") '''
        world_z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(world_z, v_norm)
        if np.linalg.norm(axis) < 1e-6:
            q_vec = pyq.Quaternion()  # identity if already aligned
        else:
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(world_z, v_norm))
            q_vec = pyq.Quaternion(axis=axis, angle=angle)
         # log every 5 seconds
        '''now = time.time()
        if now - self._last_log_time >= 5.0:
            self.get_logger().info(
                f"Injection‐vector quaternion → "
                f"w={q_vec.w:.4f}, x={q_vec.x:.4f}, "
                f"y={q_vec.y:.4f}, z={q_vec.z:.4f}"
            )
            self._last_log_time = now '''

        #publish all the markers
        self.publish_eye_and_arrow(eyeball, iris, pupil, arrow)


def main(args=None):
    rclpy.init(args=args)
    node = PosePrinter()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
