#!/usr/bin/env python3
import math, time, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Pose, Point
from tf2_ros import Buffer, TransformListener, TransformException
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene
from moveit_msgs.msg import CollisionObject, PlanningScene, RobotState
from shape_msgs.msg import Mesh, MeshTriangle

def slerp(q0, q1, t):
    """q0/q1 are [w,x,y,z]; return normalized [w,x,y,z]."""
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        q = q0 + t*(q1 - q0)
        n = np.linalg.norm(q)
        return q / (n if n > 0 else 1.0)
    theta_0 = math.acos(max(min(dot, 1.0), -1.0))
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta)/sin_0
    s1 = math.sin(theta)/sin_0
    q = s0*q0 + s1*q1
    n = np.linalg.norm(q)
    return q / (n if n > 0 else 1.0)

class InjectionPathPlanner(Node):
    def __init__(self):
        super().__init__('injection_path_planner')

        # -------- Parameters --------
        self.declare_parameter('planning_group', 'ur_manipulator')
        self.declare_parameter('ik_link', 'tool0')
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('stream_rate_hz', 100.0)
        self.declare_parameter('linear_step', 0.0015)
        self.declare_parameter('max_step', 0.001)  # MoveIt resamples at this; we’ll clamp to linear_step
        self.declare_parameter('jump_threshold', 0.0)
        self.declare_parameter('avoid_collisions', True)
        self.declare_parameter('use_moveit_validation', True)
        self.declare_parameter('gcp_service_name', '/compute_cartesian_path')
        self.declare_parameter('ps_service_name',  '/apply_planning_scene')
        self.declare_parameter('fsm_topic', 'fsm_target_frame')
        self.declare_parameter('controller_topic', 'target_frame')
        self.declare_parameter('debug', True)

        # Camera FOV keep-out
        self.declare_parameter('add_camera_fov_obstacle', True)
        self.declare_parameter('camera_fov_id', 'camera_fov_pyramid')
        self.declare_parameter('cam_tip',  [-0.016, -0.145, 0.372])
        self.declare_parameter('cam_b0',   [-0.127,  0.150, 0.230])
        self.declare_parameter('cam_b1',   [-0.127,  0.150, 0.410])
        self.declare_parameter('cam_b2',   [ 0.093,  0.150, 0.230])
        self.declare_parameter('cam_b3',   [ 0.093,  0.150, 0.410])
        self.declare_parameter('cam_fov_margin', 0.0)

        # -------- Read params --------
        gp = self.get_parameter
        self.group   = gp('planning_group').get_parameter_value().string_value
        self.ik_link = gp('ik_link').get_parameter_value().string_value
        self.base    = gp('planning_frame').get_parameter_value().string_value
        self.rate_hz = gp('stream_rate_hz').get_parameter_value().double_value
        self.linear_step = gp('linear_step').get_parameter_value().double_value
        self.max_step_cfg = gp('max_step').get_parameter_value().double_value
        self.jump_threshold_cfg = gp('jump_threshold').get_parameter_value().double_value
        self.avoid_collisions_cfg = gp('avoid_collisions').get_parameter_value().bool_value
        self.use_moveit_desired = gp('use_moveit_validation').get_parameter_value().bool_value
        self.gcp_name = gp('gcp_service_name').get_parameter_value().string_value
        self.ps_name  = gp('ps_service_name').get_parameter_value().string_value
        self.fsm_topic= gp('fsm_topic').get_parameter_value().string_value
        self.out_topic= gp('controller_topic').get_parameter_value().string_value
        self.debug    = gp('debug').get_parameter_value().bool_value

        self.add_fov  = gp('add_camera_fov_obstacle').value
        self.fov_id   = gp('camera_fov_id').value
        self.cam_tip  = np.array(gp('cam_tip').value, dtype=float)
        self.cam_b0   = np.array(gp('cam_b0').value, dtype=float)
        self.cam_b1   = np.array(gp('cam_b1').value, dtype=float)
        self.cam_b2   = np.array(gp('cam_b2').value, dtype=float)
        self.cam_b3   = np.array(gp('cam_b3').value, dtype=float)
        self.cam_margin = float(gp('cam_fov_margin').value)

        # -------- I/O --------
        self.sub = self.create_subscription(PoseStamped, self.fsm_topic, self.on_goal, 10)
        self.pub = self.create_publisher(PoseStamped, self.out_topic, 10)

        # -------- TF --------
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        # -------- MoveIt clients + topic fallback --------
        self.gcp_cli = self.create_client(GetCartesianPath, self.gcp_name)
        self.ps_cli  = self.create_client(ApplyPlanningScene, self.ps_name)
        # Transient Local so a late-joining PlanningSceneMonitor still gets the diff
        self.ps_pub  = self.create_publisher(
            PlanningScene, 'planning_scene',
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
        )

        # -------- State --------
        self.current_stream_id = 0
        self.stream = []
        self.timer = None
        self.idx = 0
        self.last_goal_time = None
        self.last_goal_dist = None
        self.scene_ready = False

        # -------- Timers --------
        self.create_timer(1.0, self._scene_tick)
        self.create_timer(2.0, self._diagnostics_tick)
        self.create_timer(3.0, self._scan_services_tick)

        self.get_logger().info(
            f"Planner ready. Subscribed to '{self.fsm_topic}', publishing to '{self.out_topic}'. "
            f"Base='{self.base}', IK link='{self.ik_link}', group='{self.group}'. "
            f"GCP='{self.gcp_name}', PS='{self.ps_name}'."
        )

    # ---------- Service scanner ----------
    def _scan_services_tick(self):
        names_and_types = self.get_service_names_and_types()
        gcp_candidates = [n for n, _ in names_and_types
                          if n.endswith('/compute_cartesian_path') or n == 'compute_cartesian_path']
        ps_candidates  = [n for n, _ in names_and_types
                          if n.endswith('/apply_planning_scene') or n == 'apply_planning_scene']

        def maybe_switch(current_name, candidates, label):
            if current_name in candidates:
                return current_name
            if self.debug:
                self.get_logger().warn(
                    f"Configured {label}='{current_name}' not found; candidates:\n  - "
                    + ("\n  - ".join(candidates) if candidates else "(none)")
                )
            if len(candidates) == 1:
                new_name = candidates[0]
                self.get_logger().info(f"Auto-selecting {label}='{new_name}'.")
                return new_name
            return current_name

        new_gcp = maybe_switch(self.gcp_name, gcp_candidates, 'gcp_service_name')
        new_ps  = maybe_switch(self.ps_name,  ps_candidates,  'ps_service_name')

        if new_gcp != self.gcp_name:
            self.gcp_name = new_gcp
            self.gcp_cli = self.create_client(GetCartesianPath, self.gcp_name)
        if new_ps != self.ps_name:
            self.ps_name = new_ps
            self.ps_cli  = self.create_client(ApplyPlanningScene, self.ps_name)

    # ---------- Diagnostics ----------
    def _diagnostics_tick(self):
        try:
            latest = rclpy.time.Time()  # TF2 "latest" (time=0), not wall-clock now
            tf_ok = self.tf_buf.can_transform(self.base, self.ik_link, latest, RclpyDuration(seconds=0.1))
        except Exception:
            tf_ok = False
        pub_cnt = self.pub.get_subscription_count()
        try:
            sub_srcs = self.sub.get_publisher_count()
        except Exception:
            sub_srcs = -1
        last_goal_age = (time.time() - self.last_goal_time) if self.last_goal_time else None
        moveit_flag = 'ON' if self.use_moveit_desired else 'OFF'
        msg = (f"[HB] TF {self.base}->{self.ik_link}: {'OK' if tf_ok else 'MISSING'} | "
               f"FSM pubs '{self.fsm_topic}': {sub_srcs} | "
               f"Controller subs '{self.out_topic}': {pub_cnt} | "
               f"MoveIt(desired): {moveit_flag} | Scene: {'READY' if self.scene_ready else 'PENDING'} | ")
        msg += f"last goal {last_goal_age:.1f}s ago, dist={self.last_goal_dist:.4f} m" if last_goal_age is not None else "no goal received yet"
        self.get_logger().info(msg)

    # ---------- Camera FOV object ----------
    def _inflate(self, p, tip=None, k=0.0):
        if k <= 0.0 or tip is None: return p
        v = p - tip; n = np.linalg.norm(v)
        return p if n <= 1e-9 else p + (k/n)*v

    def _make_pyramid_mesh(self, tip, b0, b1, b2, b3):
        mesh = Mesh()
        for v in [tip, b0, b1, b2, b3]:
            mesh.vertices.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))
        def tri(a,b,c):
            t = MeshTriangle(); t.vertex_indices[:] = [a,b,c]; return t
        mesh.triangles = [
            tri(1,3,2), tri(2,3,4),      # base
            tri(0,1,2), tri(0,2,4),      # sides
            tri(0,4,3), tri(0,3,1),
        ]
        return mesh

    def _identity_pose(self):
        p = Pose(); p.orientation.w = 1.0
        return p

    def _scene_tick(self):
        if self.scene_ready or not self.add_fov:
            return

        tip = self.cam_tip
        b0  = self._inflate(self.cam_b0, tip, self.cam_margin)
        b1  = self._inflate(self.cam_b1, tip, self.cam_margin)
        b2  = self._inflate(self.cam_b2, tip, self.cam_margin)
        b3  = self._inflate(self.cam_b3, tip, self.cam_margin)
        mesh = self._make_pyramid_mesh(tip, b0, b1, b2, b3)

        co = CollisionObject()
        co.header.frame_id = self.base
        co.id = self.fov_id
        co.meshes = [mesh]
        co.mesh_poses = [self._identity_pose()]
        co.operation = CollisionObject.ADD

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)

        # Prefer service, but don't block if it's not up yet
        used_service = False
        if not self.ps_cli.service_is_ready():
            self.ps_cli.wait_for_service(timeout_sec=0.2)
        if self.ps_cli.service_is_ready():
            req = ApplyPlanningScene.Request(scene=ps)
            future = self.ps_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            resp = future.result()
            if resp is not None and getattr(resp, "success", True):
                self.scene_ready = True
                self.get_logger().info(
                    f"Applied camera FOV collision object '{self.fov_id}' via '{self.ps_name}' (frame '{self.base}').")
                used_service = True
            else:
                # fall through to topic
                if resp is not None and hasattr(resp, "success") and not resp.success:
                    self.get_logger().warn(f"'{self.ps_name}' responded success=False; falling back to topic.")
                else:
                    self.get_logger().warn(f"'{self.ps_name}' timed out or returned None; falling back to topic.")

        if not used_service:
            # Fallback: publish diff on /planning_scene topic (Transient Local QoS)
            self.ps_pub.publish(ps)
            self.scene_ready = True  # TL ensures late subscribers get it
            self.get_logger().warn(
                "Published PlanningScene diff on '/planning_scene' (Transient Local).")

    # ---------- Goal handling ----------
    def on_goal(self, msg: PoseStamped):
        self.last_goal_time = time.time()
        self.current_stream_id += 1
        if self.timer: self.timer.cancel()

        # Current EE pose (TF2 "latest" query)
        try:
            latest = rclpy.time.Time()  # time=0 means "latest"
            self.tf_buf.can_transform(self.base, self.ik_link, latest, RclpyDuration(seconds=0.5))
            tf = self.tf_buf.lookup_transform(self.base, self.ik_link, latest)
        except TransformException as ex:
            self.get_logger().error(f"TF {self.base}->{self.ik_link} failed: {ex}")
            return

        p0 = np.array([tf.transform.translation.x,
                       tf.transform.translation.y,
                       tf.transform.translation.z], dtype=float)
        # NOTE: we keep quaternions internally as [w,x,y,z] for slerp clarity.
        q0 = np.array([tf.transform.rotation.w, tf.transform.rotation.x,
                       tf.transform.rotation.y, tf.transform.rotation.z], dtype=float)
        p1 = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        q1 = np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                       msg.pose.orientation.y, msg.pose.orientation.z], dtype=float)

        dist = float(np.linalg.norm(p1 - p0))
        self.last_goal_dist = dist
        n = max(2, int(math.ceil(dist / self.linear_step)))
        ts = np.linspace(0.0, 1.0, n)

        if self.debug:
            self.get_logger().info(
                f"Received goal from '{self.fsm_topic}': dist={dist:.4f} m, waypoints={n}, "
                f"linear_step={self.linear_step:.4f} m."
            )

        # Build waypoints
        waypoints = []
        for t in ts:
            p = (1.0 - t) * p0 + t * p1
            q = slerp(q0, q1, t)  # normalized [w,x,y,z]
            ps = Pose()
            ps.position.x, ps.position.y, ps.position.z = p.tolist()
            # geometry_msgs uses (x,y,z,w); we stored [w,x,y,z]
            ps.orientation.w, ps.orientation.x, ps.orientation.y, ps.orientation.z = q.tolist()
            waypoints.append(ps)

        # (Optional) local guardrail – prevents entering the FOV even if scene is late
        # waypoints = self._truncate_if_inside_fov(waypoints)

        # MoveIt validation (try; fallback on timeout)
        if self.use_moveit_desired and len(waypoints) >= 2:
            req = GetCartesianPath.Request()
            req.header.frame_id = self.base
            req.header.stamp = self.get_clock().now().to_msg()
            req.group_name = self.group
            req.link_name = self.ik_link
            req.waypoints = waypoints
            # Keep MoveIt sampling step consistent with our interpolation density
            req.max_step = float(min(self.max_step_cfg, self.linear_step))
            req.jump_threshold = float(self.jump_threshold_cfg)
            req.avoid_collisions = bool(self.avoid_collisions_cfg)
            # Be explicit about "start from current"
            req.start_state = RobotState()
            req.start_state.is_diff = True

            if self.debug:
                self.get_logger().info(
                    f"Calling '{self.gcp_name}' with {len(waypoints)} waypoints "
                    f"(max_step={req.max_step:.4f}, avoid_collisions={req.avoid_collisions})."
                )

            if not self.gcp_cli.service_is_ready():
                self.gcp_cli.wait_for_service(timeout_sec=0.2)

            if self.gcp_cli.service_is_ready():
                future = self.gcp_cli.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                resp = future.result()
            else:
                resp = None

            if resp is None:
                self.get_logger().warn(f"'{self.gcp_name}' unavailable/timeout. Streaming UNVALIDATED path.")
            else:
                if resp.error_code.val != 1 or resp.fraction <= 0.0:
                    self.get_logger().warn(
                        f"MoveIt rejected path (code={resp.error_code.val}, fraction={resp.fraction:.3f}). "
                        f"NOT streaming; check collisions/IK.")
                    return
                if resp.fraction < 0.999:
                    valid_count = max(2, int(math.floor(resp.fraction * len(waypoints))))
                    self.get_logger().warn(
                        f"MoveIt truncated path at fraction={resp.fraction:.3f}. "
                        f"Streaming first {valid_count}/{len(waypoints)} waypoints.")
                    waypoints = waypoints[:valid_count]

        # Stream
        out = []
        for ps in waypoints:
            stamped = PoseStamped()
            stamped.header.frame_id = self.base
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.pose = ps
            out.append(stamped)

        self.stream = out
        self._start_stream(self.current_stream_id)

    # ---------- Optional local guardrail ----------
    def _truncate_if_inside_fov(self, waypoints):
        """Stop at the first waypoint that enters the pyramid (tip->base)."""
        tip = self.cam_tip
        b = [self.cam_b0, self.cam_b1, self.cam_b2, self.cam_b3]
        base_center = sum(b) / 4.0

        # Base plane normal pointing from base toward tip
        n_base = np.cross(self.cam_b2 - self.cam_b0, self.cam_b1 - self.cam_b0)
        if np.dot(n_base, self.cam_tip - self.cam_b0) < 0:  # make it point to tip
            n_base = -n_base
        s_tip = np.dot(n_base, self.cam_tip - self.cam_b0)

        # Four side plane normals outward (use (tip, bi, b_{i+1}))
        sides = []
        order = [0,2,3,1]  # matches mesh base winding
        for i in range(4):
            a = tip
            bi = b[order[i]]
            bj = b[order[(i+1) % 4]]
            n = np.cross(bi - a, bj - a)  # normal
            # Ensure normal points OUT of pyramid (base_center should be inside -> negative side)
            if np.dot(n, base_center - a) > 0:
                n = -n
            sides.append((a, n))

        safe = []
        for ps in waypoints:
            p = np.array([ps.position.x, ps.position.y, ps.position.z])
            # between base plane and tip
            s = np.dot(n_base, p - self.cam_b0)
            inside_axial = (0.0 <= s <= s_tip + 1e-6)
            # inside all four side half-spaces (<= 0 means outside; >0 means inside)
            inside_sides = all(np.dot(n, p - a) <= 0.0 + 1e-6 for (a, n) in sides)
            if inside_axial and inside_sides:
                self.get_logger().warn("Waypoint enters camera FOV; truncating locally before MoveIt validation.")
                break
            safe.append(ps)
        return safe if len(safe) >= 2 else waypoints[:1]  # keep at least one to avoid empty stream

    # ---------- Streaming ----------
    def _start_stream(self, stream_id: int):
        period = 1.0 / max(self.rate_hz, 1.0)
        self.idx = 0
        if self.timer: self.timer.cancel()
        self.timer = self.create_timer(period, lambda: self._tick(stream_id))
        self.get_logger().info(
            f"Streaming {len(self.stream)} poses to '{self.out_topic}' at {self.rate_hz:.1f} Hz "
            f"(can preempt on new goal)."
        )

    def _tick(self, stream_id: int):
        if stream_id != self.current_stream_id:
            return
        if self.idx >= len(self.stream):
            self.timer.cancel()
            self.get_logger().info("Stream complete.")
            return
        msg = self.stream[self.idx]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.idx += 1

def main():
    rclpy.init()
    node = InjectionPathPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
