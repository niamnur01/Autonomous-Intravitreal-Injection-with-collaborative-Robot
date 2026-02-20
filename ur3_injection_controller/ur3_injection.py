#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from enum import Enum, auto

from project_interfaces.srv import Pose
from geometry_msgs.msg import PoseStamped
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException

import numpy as np
import pyquaternion as pyq
import transforms3d.euler as eul
import transforms3d.quaternions as quat

from std_srvs.srv import Trigger
import json


DATASET_NUMBER_STD_DEVIATION = 40
STABILITY_THRESHOLD = 0.002

EYE_RADIUS = 0.015
SAFE_DISTANCE = 0.03
INJECTION_DEPTH = 0.005

class Phase(Enum):
    APPROACH = auto()
    INJECT   = auto()
    POSITION = auto()
    DEPLOY   = auto()
    RETRACT  = auto()
    DISPLACE = auto()
    HOME     = auto()
    IDLE     = auto()

class Ur3_controller(Node):
    def __init__(self):
        # Initializing class
        super().__init__('ur3_controller')

        # fetch eye center from /get_eye_centers
        self.get_logger().info("Calling /get_eye_centers…")
        eye_cli = self.create_client(Trigger, '/get_eye_centers')
        while not eye_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /get_eye_centers…")
        trig_req = Trigger.Request()
        future = eye_cli.call_async(trig_req)
        rclpy.spin_until_future_complete(self, future)
        trig_resp = future.result()
        if not (trig_resp and trig_resp.success):
            self.get_logger().error("Could not fetch eye centers; shutting down.")
            rclpy.shutdown()
            return

        centers = json.loads(trig_resp.message)
        # hardcode choice for now:
        chosen = centers['right_center']
        # store as numpy array for all later use
        self.target_position = np.array(chosen)
        self.get_logger().info(f"Using eye center: {self.target_position}")
   
        # Creating client(eye-tracking) to get pose
        self.cli = self.create_client(Pose, 'gazeSrv')
        # keep the last eye-pose here:
        self.latest_eye_q = np.array([0.,0.,0.,0.])
        self.variation_buffer = []
        self.stability_check = False

        # hold the future so it doesn’t get garbage-collected
        self._eye_future = None

        # Waiting for eye-tracking to start
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Pose.Request()

        # Creating publisher to publish planning
        self.publisher = self.create_publisher(PoseStamped, 'target_frame', 10)

        #end effector pose buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Getting current configuration 
        self.homing_pose = self.get_pose()

        ##Define vector rotation for injection trajectory, prepare rotation matrix
        yaw_deg = 100 # or any value within [-80, 80] for left eye, and [100, 260] for right eye
        pitch_deg = 45.5 # or a sample within [44.5, 46.5]
        R_pitch = eul.euler2mat(0, np.deg2rad(pitch_deg), 0, 'sxyz')
        R_yaw = eul.euler2mat(0, 0, np.deg2rad(yaw_deg), 'sxyz')
        self.R_inject = R_yaw @ R_pitch

        # Periodically collect eye orientation at 10Hz to check stability
        self.gaze_timer = self.create_timer(0.1, self.poll_eye_orientation)
        self.get_logger().info("Waiting for proper eye orientation before starting FSM…")
        self.waiting_for_stable_eye = True
        # Wait for stability before proceeding
        self.eye_check_timer = self.create_timer(0.2, self.eye_ready_check)

        # Needle orientation for stable alignement through injection and retraction
        self.sequence_inj_point = None
        self.sequence_q_tool = None
        self.sequence_tool_z = None

        # create periodic FSM driver (50Hz)
        self.timer = self.create_timer(1.0/50.0, self._fsm_step)

        #New injection cycle control
        self.waiting_for_user = False
        self.user_input_timer = self.create_timer(1.0, self.check_user_input)

        # --- DEPLOY phase state (non-blocking timer) ---
        self.deploy_timer = None
        self.deploy_done = False
        
    def _fsm_step(self):
        if self.waiting_for_stable_eye:
            return

        if self.phase == Phase.IDLE:
            self.get_logger().info('Phase: APPROACH -> sending approach target')
            self.phase = Phase.APPROACH

        if self.phase == Phase.APPROACH:
            if not self.phase_sent[Phase.APPROACH]:
                self.motion_approaching_target()
                self.phase_sent[Phase.APPROACH] = True
            elif self.at_target(self.approach_target):
                self.get_logger().info('Approach reached; moving to POSITION')
                self.phase = Phase.POSITION
        
        elif self.phase == Phase.POSITION:
            if not self.phase_sent[Phase.POSITION]:
                self.motion_eye_retraction()
                self.phase_sent[Phase.POSITION] = True
            elif self.at_target(self.retract_target):
                self.get_logger().info('Ready to inject; moving to INJECT')
                self.phase = Phase.INJECT

        elif self.phase == Phase.INJECT:
            if not self.phase_sent[Phase.INJECT]:
                self.motion_eye_injection()
                self.phase_sent[Phase.INJECT] = True
            else:
               _ = self.get_eye_orientation(0)
               if not self.stability_check:
                   self.get_logger().warn('Eye moved—aborting injection, retracting!')
                   self.phase = Phase.RETRACT
                   return #avoid checking at_target
            if self.at_target(self.injection_target):
                self.get_logger().info('Injection reached; moving to DEPLOY')
                self.phase = Phase.DEPLOY
        
        elif self.phase == Phase.DEPLOY:
            if not self.phase_sent[Phase.DEPLOY]:
                self.get_logger().info('Phase: DEPLOY -> holding for 2.5s')
                self.start_deploy_timer(2.5)
                self.phase_sent[Phase.DEPLOY] = True
            elif self.deploy_done:
                self.get_logger().info('DEPLOY timer elapsed; moving to RETRACT')
                self.phase = Phase.RETRACT

        elif self.phase == Phase.RETRACT:
            if not self.phase_sent[Phase.RETRACT]:
                self.motion_eye_retraction()
                self.phase_sent[Phase.RETRACT] = True
            elif self.at_target(self.retract_target):
                self.get_logger().info('Exited eye; moving to DISPLACE')
                self.phase = Phase.DISPLACE
        
        elif self.phase == Phase.DISPLACE:
            if not self.phase_sent[Phase.DISPLACE]:
                self.motion_approaching_target()
                self.phase_sent[Phase.DISPLACE] = True
            elif self.at_target(self.approach_target):
                self.get_logger().info('Tool displaced; moving to HOME')
                self.phase = Phase.HOME

        elif self.phase == Phase.HOME:
            if not self.phase_sent[Phase.HOME]:
                self.homing_procedure()
                self.phase_sent[Phase.HOME] = True
                self.get_logger().info('Homing reached; sequence DONE')
            elif self.at_target(self.homing_pose):
                self.waiting_for_user = True


    def eye_ready_check(self):
        if self.waiting_for_stable_eye and self.stability_check and self.valid_gaze_range(self.latest_eye_q):
            self.get_logger().info("Stable eye orientation detected — restarting FSM.")
            self.stability_check = False
            self.phase_sent = {p: False for p in Phase}
            if self.deploy_timer is not None:
                try:
                    self.deploy_timer.cancel()
                except Exception:
                    pass
                self.deploy_timer = None
            self.deploy_done = False
            self.waiting_for_stable_eye = False
            self.phase = Phase.IDLE

    def motion_approaching_target(self): 
        # 1) Get a neutral approach orientation / injection point
        eye_pos    = self.target_position
        eye_ori    = self.get_eye_orientation(0)
        inj_pt, ori = self.compute_injection_pose_from_eye_frame(eye_pos, eye_ori)

        # 2) Step back along tool Z axis by safe_distance
        tool_z = np.array([
            2 * (ori[1]*ori[3] + ori[0]*ori[2]),
            2 * (ori[2]*ori[3] - ori[0]*ori[1]),
            1 - 2*(ori[1]**2 + ori[2]**2)
        ])
        approach_pt = inj_pt - SAFE_DISTANCE * tool_z

        self.sequence_q_tool = ori
        self.sequence_inj_point = inj_pt
        self.sequence_tool_z = tool_z

        # 3) Plan & publish trajectory from current_pose → approach_pt
        self.approach_target = np.concatenate((approach_pt, ori))
        self.publish_pose(approach_pt, ori)

    def motion_eye_injection(self):
        inj_point = self.sequence_inj_point
        inj_orientation = self.sequence_q_tool
        tool_z = self.sequence_tool_z

        position = inj_point + INJECTION_DEPTH * tool_z

        # Compute and execute trajectory
        self.injection_target = np.concatenate((position, inj_orientation))
        self.publish_pose(position, inj_orientation)

    def motion_eye_retraction(self):
        inj_point = self.sequence_inj_point
        inj_orientation = self.sequence_q_tool
        tool_z = self.sequence_tool_z
        
        position = inj_point - INJECTION_DEPTH * tool_z

        # Compute and execute trajectory
        self.retract_target = np.concatenate((position, inj_orientation))
        self.publish_pose(position, inj_orientation)
    
    def homing_procedure(self):
        self.publish_pose(self.homing_pose[:3], self.homing_pose[3:])

    def get_pose(self):
        waiting_pose = True
        # Waiting to read the current pose 
        while waiting_pose:           
            try:
                future = self.tf_buffer.wait_for_transform_async(
                        'base_link',
                        'tool0',
                        rclpy.time.Time()
                        )
                rclpy.spin_until_future_complete(self, future)
                trans = self.tf_buffer.lookup_transform(
                    'base_link',
                    'tool0',
                    rclpy.time.Time()
                    )
                return  np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z,
                                    trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z])
            except TransformException as ex:
                self.get_logger().info(f'Could not transform: {ex}')

    def compute_injection_pose_from_eye_frame(self, eye_position, eye_orientation, radius=0.015):
        # Compute pre-rotation quaternion based on eye position to set the spawning eye orientation
        q_trans = eul.euler2quat(np.pi, 0, -np.pi/2, 'rzyx')  # returns w,x,y,z
        # Combined quaternion for eye gaze and eye frame rotation (for injection_vector)
        w, x, y, z = self.quaternion_multiply(q_trans, eye_orientation)
        #Create the vector injection trajectory, and rotate it into the world
        v_rotated = self.R_inject @ np.array([0, 0, EYE_RADIUS])
        q_frame = [w, x, y, z]
        R_frame = quat.quat2mat(q_frame)
        v_world = R_frame @ v_rotated
        #injection point
        position_world = eye_position + v_world

        #Relate injection vector to needle vector q_tool
        z_axis = eye_position - position_world
        z_axis /= np.linalg.norm(z_axis)
        x_guess = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.9 else np.array([0, 1, 0])
        x_axis = np.cross(x_guess, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_tool = np.column_stack((x_axis, y_axis, z_axis))
        q_tool = quat.mat2quat(R_tool)

        '''# Log all intermediate values
        self.get_logger().info(f"""
        ==== Injection Pose Debug Info ====
        eye_position: {eye_position}
        eye_orientation (input): {eye_orientation}
        q_trans: {q_trans}
        q_frame (eye_orientation * q_trans): {q_frame}
        v_rotated (in eye frame): {v_rotated}
        R_frame (from q_frame):\n{R_frame}
        v_world (in world frame): {v_world}
        position_world (injection point): {position_world}
        z_axis (needle dir): {z_axis}
        x_axis: {x_axis}
        y_axis: {y_axis}
        R_tool (rotation matrix):\n{R_tool}
        q_tool (tool orientation): {q_tool}
        ====================================""")  '''
        return position_world, q_tool

    def stability_control(self, orientation):
        length = len(self.variation_buffer)

        if length == 0:
            self.variation_buffer = np.array([orientation])
        elif length < DATASET_NUMBER_STD_DEVIATION:
            self.variation_buffer = np.append(self.variation_buffer, [orientation], axis=0)
        else:
            self.variation_buffer = np.delete(self.variation_buffer, 0, axis=0)
            self.variation_buffer = np.append(self.variation_buffer, [orientation], axis=0)

            std = np.std(self.variation_buffer, ddof=1, axis=0)
            std_mean = np.mean(std)
            #self.get_logger().info(f"Stability check: std_mean = {std_mean:.6f} → check = {std_mean < STABILITY_THRESHOLD}")
            self.stability_check = (std_mean < STABILITY_THRESHOLD)

    def get_eye_orientation(self, request):
        # fire off a new request, but don’t wait
        self.req.r = request
        self._eye_future = self.cli.call_async(self.req)
        self._eye_future.add_done_callback(self.on_eye_orientation)
        # return the last known orientation (might be a little stale)
        return self.latest_eye_q
    
    def on_eye_orientation(self, future):
        # this is called *inside* the executor, non-blocking
        try:
            resp = future.result()
            if resp is None:
                self.get_logger().warn("Eye orientation service returned None.")
                return

            q = np.array([resp.w, resp.x, resp.y, resp.z])
            if np.allclose(q, np.zeros(4)):
                self.get_logger().warn("Received zero quaternion — ignoring.")
                return

            self.latest_eye_q = q
            self.stability_control(q)

        except Exception as e:
            self.get_logger().error(f"Exception in eye orientation callback: {e}")

    def poll_eye_orientation(self):
        _ = self.get_eye_orientation(0)

    def valid_gaze_range(self, q):
        if np.allclose(q, np.zeros(4)):
            return False
        '''Ranges for right eye, left should be specular
        # Extreme Up-Left (toward root of the nose)
            x ∈ [0.105, 0.130]
            y ∈ [-0.27, -0.24]
        # Extreme Up-Right
            x ∈ [-0.20, -0.17]
            y ∈ [-0.11, -0.08]
        # Extreme Down-Left
            x ∈ [0.050, 0.085]
            y ∈ [0.253, 0.268]'''
        x, y = q[1], q[2] #assuming q=[w,x,y,z]
        return (0.07<= x  and y >=-0.110)

    
    def publish_pose(self, position, orientation):
        msg = PoseStamped()

        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]

        msg.pose.orientation.w = orientation[0]
        msg.pose.orientation.x = orientation[1]
        msg.pose.orientation.y = orientation[2]
        msg.pose.orientation.z = orientation[3]
        
        self.current_pose = np.concatenate((position, orientation))
        self.get_logger().info(f"Publishing {position}, {orientation}")
        self.publisher.publish(msg)

    def quaternion_multiply(self, quaternion1, quaternion0):     
        q1 = pyq.Quaternion(quaternion1)
        q0 = pyq.Quaternion(quaternion0)
        q = q1*q0

        return np.array([q.w, q.x, q.y, q.z])

    def at_target(self, target, pos_tol=0.001, ori_tol=0.01):
        actual = self.get_pose()
        pos_err = np.linalg.norm(actual[:3] - target[:3])

        # --- Orientation error (sign-invariant) ---
        qa = actual[3:].astype(float)
        qt = target[3:].astype(float)
        # normalize (required so the dot gives a true rotation angle)
        n_qa = np.linalg.norm(qa); n_qt = np.linalg.norm(qt)
        if not np.isfinite(n_qa) or not np.isfinite(n_qt) or n_qa < 1e-12 or n_qt < 1e-12:
            return False
        qa /= n_qa
        qt /= n_qt
        # sign-invariant via abs(dot)
        dot = float(np.clip(abs(np.dot(qa, qt)), -1.0, 1.0))
        ori_err = 2.0 * np.arccos(dot)   # radians in [0, pi]

        return (pos_err < pos_tol) and (ori_err < ori_tol)
    
    def check_user_input(self):
        if self.waiting_for_user:
            print("\n>>> Press [R] to run another injection or [Q] to quit:")
            choice = input().strip().lower()
            if choice == 'r':
                self.get_logger().info("Restarting injection sequence.")
                self.variation_buffer = []
                self.stability_check = False
                self.latest_eye_q = np.array([0., 0., 0., 0.])
                self.waiting_for_stable_eye = True    
                self.waiting_for_user = False
                self.phase_sent = {p: False for p in Phase}
                if self.deploy_timer is not None:
                    try:
                        self.deploy_timer.cancel()
                    except Exception:
                        pass
                    self.deploy_timer = None
                self.deploy_done = False
                self.phase = Phase.IDLE
            elif choice == 'q':
                self.get_logger().info("Shutting down node.")
                rclpy.shutdown()

    def start_deploy_timer(self, seconds: float = 2.5):
        # Cancel any previous timer (defensive)
        if self.deploy_timer is not None:
            try:
                self.deploy_timer.cancel()
            except Exception:
                pass
        self.deploy_done = False
        # create_timer returns a periodic timer; we'll cancel it in the callback
        self.deploy_timer = self.create_timer(seconds, self.on_deploy_done)

    def on_deploy_done(self):
        # Timer fired once → mark done and cancel so it doesn't repeat
        self.deploy_done = True
        if self.deploy_timer is not None:
            try:
                self.deploy_timer.cancel()
            except Exception:
                pass
            self.deploy_timer = None


    
def main():
    rclpy.init()

    ur3_controller = Ur3_controller()
    rclpy.spin(ur3_controller)

    ur3_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
