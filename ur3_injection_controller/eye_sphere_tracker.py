#!/usr/bin/env python3
import math
import json
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

class EyeSphereTracker(Node):
    def __init__(self):
        #Near-far clipping plane of the coppeliasim depth sensor. Usend to un-normalize the depth measures    
        self.near = 0.50  # m
        self.far  = 0.80  # m

        super().__init__('eye_sphere_tracker')
        self.br = CvBridge()

        # Subscribers for synced RGB + depth
        self.rgb_sub   = Subscriber(self, Image, '/vision_rgb')
        self.depth_sub = Subscriber(self, Image, '/vision_depth')
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self._calib_cb)

        # FaceMesh setup
        mp_fm = mp.solutions.face_mesh
        self.face_mesh = mp_fm.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark indices per eye
        self.left_idxs  = [157,158,159,160,153,145,144,163]
        self.right_idxs = [384,385,386,387,390,380,374,373]

        # Camera intrinsics
        self.W = self.H = 1080
        vfov_rad = math.radians(33.78)
        self.fx = self.fy = (self.W/2)/math.tan(vfov_rad/2)
        self.cx = self.W/2
        self.cy = self.H/2

        # World transform
        R = np.array([[-1,0,0],[0,0,-1],[0,-1,0]])
        self.R_to_world = R.T
        self.T = np.array([-0.0161, -0.14575, 0.42239])

        # Calibration storage
        self._need_frames = 50
        self._count = 0
        self._left_centers = []
        self._right_centers = []
        self._left_rs = []
        self._right_rs = []
        self._calibrated = False

        self.get_logger().info(f"Calibrating eyes: collecting {self._need_frames} frames...")

    def _collect_points(self, lm, depth):
        h, w = depth.shape[:2]
        def to_pts(idxs):
            pts = []
            for i in idxs:
                u = int(lm[i].x * self.W)
                v = int(lm[i].y * self.H)
                ud = np.clip(int(u * w / self.W), 0, w-1)
                vd = np.clip(int(v * h / self.H), 0, h-1)
                Z = get_depth_median(self, depth, ud, vd)
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                pts.append([X,Y,Z])
            return np.array(pts)
        return to_pts(self.left_idxs), to_pts(self.right_idxs)

    def _calib_cb(self, rgb_msg, depth_msg):
        if self._calibrated:
            return

        # 1) Read frames
        color = self.br.imgmsg_to_cv2(rgb_msg, 'bgr8')
        color = cv2.flip(color, 0)
        depth = self.br.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth = cv2.flip(depth, 0)

        # 2) Detect landmarks
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return
        lm = res.multi_face_landmarks[0].landmark

        # 3) Collect and fit
        left_pts, right_pts = self._collect_points(lm, depth)
        lc = fit_sphere_center(left_pts)
        rc = fit_sphere_center(right_pts)
        # radii
        lr = np.linalg.norm(left_pts - lc[None], axis=1).mean()
        rr = np.linalg.norm(right_pts - rc[None], axis=1).mean()

        # 4) To world frame
        def cam2world(c):
            cpl = np.array([-c[0], c[1], -c[2]])
            return self.R_to_world.dot(cpl) + self.T
        lw = cam2world(lc)
        rw = cam2world(rc)

        # 5) Store
        self._left_centers.append(lw)
        self._right_centers.append(rw)
        self._left_rs.append(lr)
        self._right_rs.append(rr)
        self._count += 1
        self.get_logger().info(f"Frame {self._count}/{self._need_frames} captured")

        # 6) Finish
        if self._count >= self._need_frames:
            self._finish_calibration()

    def _finish_calibration(self):
        self._calibrated = True
        # mean values
        self.left_center = np.mean(self._left_centers, axis=0)
        self.right_center = np.mean(self._right_centers, axis=0)
        self.left_radius = float(np.mean(self._left_rs))
        self.right_radius = float(np.mean(self._right_rs))

        # log final
        self.get_logger().info("=== Calibration Done ===")
        self.get_logger().info(f"Left center (world):  {tuple(self.left_center)}")
        self.get_logger().info(f"Right center (world): {tuple(self.right_center)}")
        self.get_logger().info(f"Left radius:  {self.left_radius:.3f} m")
        self.get_logger().info(f"Right radius: {self.right_radius:.3f} m")

        # create service
        self._srv = self.create_service(
            Trigger, 'get_eye_centers', self._handle_service)
        self.get_logger().info("Service '/get_eye_centers' ready")

    def _handle_service(self, request, response):
        payload = {
            'left_center':  self.left_center.tolist(),
            'right_center': self.right_center.tolist(),
            'left_radius':  self.left_radius,
            'right_radius': self.right_radius
        }
        response.success = True
        response.message = json.dumps(payload)
        return response
    

def fit_sphere_center(points: np.ndarray) -> np.ndarray:
    """
    Given an (N×3) array of 3D points on a sphere surface,
    returns the best‐fit center (x0, y0, z0) in a least‐squares sense.
    """
    p0 = points[0]
    A = 2 * (points[1:] - p0)
    b = np.sum(points[1:]**2, axis=1) - np.sum(p0**2)
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C


def get_depth_median(self, depth_img: np.ndarray, u: int, v: int, patch: int = 1) -> float:
    """
    Spatial median filter over a (2*patch+1)^2 neighborhood of (u,v).
    Returns depth in meters.
    """
    vals = []
    h, w = depth_img.shape[:2]
    for du in range(-patch, patch + 1):
        for dv in range(-patch, patch + 1):
            uu = np.clip(u + du, 0, w - 1)
            vv = np.clip(v + dv, 0, h - 1)
            z = depth_img[vv, uu]
            if z > 0:
                vals.append(z)
    if not vals:
        return 0.0
    med = float(np.median(vals))
    
    #Un-normalize 
    med = self.near + med * (self.far - self.near)
    
    return (med * 0.001) if depth_img.dtype == np.uint16 else float(med)


def main():
    rclpy.init()
    node = EyeSphereTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
