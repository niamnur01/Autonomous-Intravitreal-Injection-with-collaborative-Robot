#!/usr/bin/env python3
"""
ROS2 node that continuously queries the Pose service, prints the returned quaternion,
pre-rotates it, and visualizes the eye frame, eyeball, iris, and pupil in RViz.

ADDED:
- Compute an injection line defined by a point (position_world) and direction (inj_vector).
- Query robot end-effector pose (tool0 in base_link) via TF2.
- Extract the end-effector z-axis from its quaternion.
- Compute angular error (between ee z-axis and inj_vector) and lateral distance of ee origin to the injection line.
- Visualize the injection line and the end-effector z-axis in RViz.
"""
import time
import numpy as np
import transforms3d.euler as euler
import transforms3d.quaternions as quat
import pyquaternion as pyq

import rclpy
from rclpy.node import Node

from project_interfaces.srv import Pose

from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import TransformException


# Constants
FREQUENCY = 20  # polling rate in Hz
TIME_STEP = 1.0 / FREQUENCY
EYE_FRAME = 'eye_frame'
PARENT_FRAME = 'base_link'
EYE_POSITION = (-0.098, 0.35274, 0.629)  # [m]
EYE_RADIUS = 0.015  # [m]
SAFE_DISTANCE = 0.03
INJECTION_DEPTH = 0.005
SIZE = 1.0  # scaling factor for markers


class PosePrinter(Node):
    def __init__(self):
        super().__init__('pose_printer')

        # for 5-second logging
        self._last_log_time = time.time()

        # TF buffer/listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create client for the Pose service
        self.cli = self.create_client(Pose, 'gazeSrv')
        # Wait until the service is available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.get_logger().info('Connected to Pose service.')

        # Prepare request
        self.req = Pose.Request()

        # Define vector rotation for injection trajectory, prepare rotation matrix for environment_building
        yaw_deg = 100  # or any value within [-80, 80] for left eye, and [100, 260] for right eye
        pitch_deg = 45.5  # or a sample within [44.5, 46.5]
        R_pitch = euler.euler2mat(0, np.deg2rad(pitch_deg), 0, 'sxyz')
        R_yaw = euler.euler2mat(0, 0, np.deg2rad(yaw_deg), 'sxyz')
        self.R_inject = R_yaw @ R_pitch

        # RViz publishers
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.current_position = np.array(EYE_POSITION, dtype=float)

        # --- File logging setup ---
        self.error_log_path = 'E_E error logs' # per user request, including spaces
        # Clear file at start
        try:
            with open(self.error_log_path, 'w') as f:
                f.write('')
        except Exception as e:
            self.get_logger().warn(f'Could not initialize error log file: {e}')

    # ---------- math helpers ----------
    def quaternion_multiply(self, q1_arr, q0_arr):
        q1 = pyq.Quaternion(q1_arr)
        q0 = pyq.Quaternion(q0_arr)
        q = q1 * q0
        return np.array([q.w, q.x, q.y, q.z])

    @staticmethod
    def safe_normalize(v, eps: float = 1e-12):
        n = np.linalg.norm(v)
        if n < eps:
            return v, n
        return v / n, n

    @staticmethod
    def angle_between(u, v, eps: float = 1e-12):
        u_n, nu = PosePrinter.safe_normalize(u, eps)
        v_n, nv = PosePrinter.safe_normalize(v, eps)
        if nu < eps or nv < eps:
            return np.nan
        dot = np.clip(np.dot(u_n, v_n), -1.0, 1.0)
        return float(np.arccos(dot))  # radians

    # ---------- robot pose via TF ----------
    def get_pose(self):
        """Return [x, y, z, w, x, y, z] of tool0 in base_link."""
        while rclpy.ok():
            try:
                future = self.tf_buffer.wait_for_transform_async(
                    PARENT_FRAME, 'tool0', rclpy.time.Time()
                )
                rclpy.spin_until_future_complete(self, future)
                trans = self.tf_buffer.lookup_transform(
                    PARENT_FRAME, 'tool0', rclpy.time.Time()
                )
                return np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                    trans.transform.rotation.w,
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                ])
            except TransformException as ex:
                self.get_logger().warn(f'Could not transform: {ex}')
                time.sleep(0.05)

    # ---------- error computation ----------
    def compute_errors(self, ee_pose, line_point, line_dir):
        """Compute (angle_error_rad, lateral_distance_m, along_projection_m, point_error_m, z_axis, cartesian_err_xyz).
        Inputs are in base_link.
        - angle_error_rad: angle between EE z-axis and line_dir.
        - lateral_distance_m: shortest distance from EE *position* to the infinite line through line_point along line_dir.
        - along_projection_m: signed projection along the line from line_point to EE position.
        - point_error_m: straight-line distance from EE origin to the target point (line_point).
        - cartesian_err_xyz: 3D vector (target - current) = line_point - p.
        """
        p = ee_pose[0:3]
        q = ee_pose[3:7] # w,x,y,z

        # EE z-axis from quaternion
        R_ee = quat.quat2mat(q)
        z_axis = R_ee[:, 2]

        # Desired direction
        d, _ = self.safe_normalize(line_dir)

        # Angular error
        ang = self.angle_between(z_axis, d)
 
       # Cartesian error (target - current)
        cart_err = line_point - p


        # For lateral distance, use r = (current - target)
        r = -cart_err
        # New along sign convention (mostly positive toward target along +d)
        along = float(np.dot(cart_err, d))
        lateral_vec = r - float(np.dot(r, d)) * d
        lateral = float(np.linalg.norm(lateral_vec))
        point_error = float(np.linalg.norm(r))

        return ang, lateral, along, point_error, z_axis, cart_err

    # ---------- main render/logic ----------
    def run(self):
        while rclpy.ok():
            # Call service
            future = self.cli.call_async(self.req)
            rclpy.spin_until_future_complete(self, future)

            if future.done() and future.result() is not None:
                resp = future.result()
                eye_q = np.array([resp.w, resp.x, resp.y, resp.z])
                self.environment_building(eye_q)
            else:
                self.get_logger().warn('Service call failed or no response.')

            time.sleep(TIME_STEP)

    def publish_markers(self, markers):
        ma = MarkerArray()
        ma.markers.extend(markers)
        self.marker_array_pub.publish(ma)

    def write_error_line(self, inj_point_world, inj_dir, ee_pos, ee_z_axis, ang_rad, lateral, along, point_err, cart_err):
        """Write one line of numeric error data with 5 decimals, space-separated, no labels.
        Order:
        inj_point_world(3), inj_dir(3), ee_pos(3), ee_z_axis(3), angle_deg, lateral, along, point_err, cart_err(3)
        """
        angle_deg = np.degrees(ang_rad) if not np.isnan(ang_rad) else np.nan
        nums = [
            *inj_point_world.tolist(),
            *inj_dir.tolist(),
            *ee_pos.tolist(),
            *ee_z_axis.tolist(),
            angle_deg, lateral, along, point_err,
            *cart_err.tolist(),
        ]
        line = ' '.join(f"{x:.5f}" for x in nums)
        try:
            with open(self.error_log_path, 'a') as f:
                f.write(line + '\n')
        except Exception as e:
            self.get_logger().warn(f'Failed writing error log line: {e}')

    def environment_building(self, eye_q):
        # Compute pre-rotation quaternion based on eye position to set the spawning eye orientation
        q_trans = euler.euler2quat(np.pi, 0, -np.pi / 2, 'rzyx')  # returns w,x,y,z
        # Combined quaternion for eye gaze and eye frame rotation (for injection_vector)
        w, x, y, z = self.quaternion_multiply(q_trans, eye_q)

        # Create the vector to visualize the injection trajectory
        # rotate the local [0, 0, 1] into the world
        v_rotated = self.R_inject @ np.array([0, 0, 1.0])
        q_frame = [w, x, y, z]
        R_frame = quat.quat2mat(q_frame)
        v_world = R_frame @ v_rotated  # direction pointing outward from eye

        # injection point (outside the eye along v_world)
        position_world = self.current_position + v_world * (EYE_RADIUS + SAFE_DISTANCE)
        # NEW: injection target point 1 cm from eye center along inward direction
        inj_point_world = self.current_position + v_world * (EYE_RADIUS - INJECTION_DEPTH)

        # NEW: direction that goes from position_world back toward the eye center
        inj_vector = self.current_position - position_world  # points inward
        inj_dir, _ = self.safe_normalize(inj_vector)

        # Publish eye frame transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = PARENT_FRAME
        t.child_frame_id = EYE_FRAME
        t.transform.translation.x = float(self.current_position[0])
        t.transform.translation.y = float(self.current_position[1])
        t.transform.translation.z = float(self.current_position[2])
        t.transform.rotation.w = float(w)
        t.transform.rotation.x = float(x)
        t.transform.rotation.y = float(y)
        t.transform.rotation.z = float(z)
        self.tf_broadcaster.sendTransform(t)

        # ---- Visual Markers ----
        markers = []

        # 1) Eyeball
        eyeball = Marker()
        eyeball.header.frame_id = EYE_FRAME
        eyeball.header.stamp = t.header.stamp
        eyeball.ns = 'eye_viz'
        eyeball.id = 0
        eyeball.type = Marker.SPHERE
        eyeball.action = Marker.ADD
        eyeball.pose.position.x = 0.0
        eyeball.pose.position.y = 0.0
        eyeball.pose.position.z = 0.0
        eyeball.pose.orientation.w = 1.0
        eyeball.scale.x = 2 * EYE_RADIUS
        eyeball.scale.y = 2 * EYE_RADIUS
        eyeball.scale.z = 2 * EYE_RADIUS
        eyeball.color.a = 0.3
        eyeball.color.r = 1.0
        eyeball.color.g = 1.0
        eyeball.color.b = 1.0
        markers.append(eyeball)

        # 2) Iris
        iris = Marker()
        iris.header = eyeball.header
        iris.ns = eyeball.ns
        iris.id = 1
        iris.type = Marker.SPHERE
        iris.action = Marker.ADD
        iris.pose.position.x = 0.0
        iris.pose.position.y = 0.0
        iris.pose.position.z = EYE_RADIUS
        iris.pose.orientation.w = 1.0
        iris.scale.x = 0.015 * SIZE
        iris.scale.y = 0.015 * SIZE
        iris.scale.z = 0.006 * SIZE
        iris.color.a = 1.0
        iris.color.r = 0.0
        iris.color.g = 150.0 / 255.0
        iris.color.b = 0.0
        markers.append(iris)

        # 3) Pupil
        pupil = Marker()
        pupil.header = eyeball.header
        pupil.ns = eyeball.ns
        pupil.id = 2
        pupil.type = Marker.SPHERE
        pupil.action = Marker.ADD
        pupil.pose.position.x = 0.0
        pupil.pose.position.y = 0.0
        pupil.pose.position.z = EYE_RADIUS + 0.003
        pupil.pose.orientation.w = 1.0
        pupil.scale.x = 0.006 * SIZE
        pupil.scale.y = 0.006 * SIZE
        pupil.scale.z = 0.0015 * SIZE
        pupil.color.a = 1.0
        pupil.color.r = 0.0
        pupil.color.g = 0.0
        pupil.color.b = 0.0
        markers.append(pupil)

        
        # 4) Injection ray from visual point to in-eye target (blue)
        ray = Marker()
        ray.header.frame_id = PARENT_FRAME
        ray.header.stamp = t.header.stamp
        ray.ns = 'injection_line'
        ray.id = 4
        ray.type = Marker.ARROW
        ray.action = Marker.ADD
        ray_start = Point(x=float(position_world[0]), y=float(position_world[1]), z=float(position_world[2]))
        ray_end_p = inj_point_world  # exactly 1 cm from eye center along inward direction
        ray_end = Point(x=float(ray_end_p[0]), y=float(ray_end_p[1]), z=float(ray_end_p[2]))
        ray.points = [ray_start, ray_end]
        ray.scale.x = 0.001
        ray.scale.y = 0.002
        ray.scale.z = 0.005
        ray.color.r = 0.0
        ray.color.g = 0.0
        ray.color.b = 1.0
        ray.color.a = 0.9
        markers.append(ray)

        # Compute errors vs. robot pose
        ee_pose = self.get_pose()
        ang, lateral, along, point_err, z_axis, cart_err = self.compute_errors(ee_pose, inj_point_world, inj_dir)

        # 6) Visualize EE z-axis at tool pose for intuition
        ee_axis = Marker()
        ee_axis.header.frame_id = PARENT_FRAME
        ee_axis.header.stamp = t.header.stamp
        ee_axis.ns = 'ee_axis'
        ee_axis.id = 5
        ee_axis.type = Marker.ARROW
        ee_axis.action = Marker.ADD
        p = ee_pose[:3]
        axis_len = 0.02
        ee_axis_end = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        ee_axis_start_p = p - z_axis * axis_len
        ee_axis_start = Point(x=float(ee_axis_start_p[0]), y=float(ee_axis_start_p[1]), z=float(ee_axis_start_p[2]))
        ee_axis.points = [ee_axis_start, ee_axis_end]
        ee_axis.scale.x = 0.002
        ee_axis.scale.y = 0.002
        ee_axis.scale.z = 0.002
        ee_axis.color.r = 0.0
        ee_axis.color.g = 1.0
        ee_axis.color.b = 0.0
        ee_axis.color.a = 0.9
        markers.append(ee_axis)

        # Publish all markers
        self.publish_markers(markers)

        # Throttled log
        now = time.time()
        if now - self._last_log_time >= 0.25: # every second
            threshold = SAFE_DISTANCE + INJECTION_DEPTH
            if point_err < threshold:
                self.get_logger().info(
                f"Target (inj_point_world): [{inj_point_world[0]:.4f}, {inj_point_world[1]:.4f}, {inj_point_world[2]:.4f}] "
                f"Injection dir (inward): [{inj_dir[0]:.4f}, {inj_dir[1]:.4f}, {inj_dir[2]:.4f}]"
                f"EE pos: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] "
                f"EE z-axis: [{z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f}] "
                f"Angle error: {np.degrees(ang):.2f} deg, Lateral-to-line: {lateral:.4f} m, Along: {along:.4f} m, Point error: {point_err:.4f} m, "
                f"Cartesian error (x,y,z): [{cart_err[0]:.4f}, {cart_err[1]:.4f}, {cart_err[2]:.4f}]"
                )
                self.write_error_line(inj_point_world, inj_dir, ee_pose[:3], z_axis, ang, lateral, along, point_err, cart_err)
            self._last_log_time = now


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
