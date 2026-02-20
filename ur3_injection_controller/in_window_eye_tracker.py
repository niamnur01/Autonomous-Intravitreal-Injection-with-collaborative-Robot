#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer


def fit_sphere_center(points: np.ndarray) -> np.ndarray:
    """
    Given an (N×3) array of 3D points on a sphere surface,
    returns the best‐fit center (x0, y0, z0) in a least‐squares sense.
    """
    p0 = points[0]
    A = 2 * (points[1:] - p0)                   # shape (N-1,3)
    b = np.sum(points[1:]**2, axis=1) - np.sum(p0**2)
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C  # (3,)


def get_depth_median(depth_img: np.ndarray, u: int, v: int, patch: int = 1) -> float:
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
    med = np.median(vals)
    # convert to meters if the raw array is uint16 (mm)
    return (med * 0.001) if depth_img.dtype == np.uint16 else float(med)


class EyeSphereTracker(Node):
    def __init__(self):
        super().__init__('eye_sphere_tracker')
        self.br = CvBridge()

        # Time‐synchronized RGB + depth
        self.rgb_sub   = Subscriber(self, Image, '/vision_rgb')
        self.depth_sub = Subscriber(self, Image, '/vision_depth')
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.cb)

        # Mediapipe FaceMesh
        mp_fm = mp.solutions.face_mesh
        self.face_mesh = mp_fm.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # The 8 corneal‐rim landmarks per eye
        self.left_idxs  = [157, 158, 159, 160, 153, 145, 144, 163, 469, 470, 471, 472]
        self.right_idxs = [384, 385, 386, 387, 390, 380, 374, 373, 474, 475, 476, 477]
        '''For future improvements:
            when refine_landmarks=True, the iris landmarks are available, landmarks id are
            Left iris ring (excluding the center): 469, 470, 471, 472
            Right iris ring (excluding the center): 474, 475, 476, 477

            This additional landmarks could prove useful when the current ones are unsufficient
            The bulge of the eye caused by the cornea starts exactly around the iris, so this 8 points shoul still belong to the spherical part of the eye
        '''

        # Camera intrinsics via FOV + resolution
        self.W, self.H = 1080, 1080
        vfov_rad = math.radians(33.78)
        self.fx = (self.W/2) / math.tan(vfov_rad/2)
        self.fy = self.fx
        self.cx = self.W/2
        self.cy = self.H/2 

        # Temporal smoothing + outlier settings
        self.smoothed_left  = None
        self.smoothed_right = None
        self.alpha          = 0.2   # EMA smoothing factor
        self.R              = 0.015  # eyeball radius in m
        self.outlier_thresh = 0.005 # 5 mm

    def cb(self, rgb_msg: Image, depth_msg: Image):
        # 1) Read & flip RGB
        color = self.br.imgmsg_to_cv2(rgb_msg, 'bgr8')
        color = cv2.flip(color, 0)

        # 2) Read & flip depth 
        depth = self.br.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth = cv2.flip(depth, 0)

        # 3) Landmark detection
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            cv2.imshow('phase3', color); cv2.waitKey(1)
            return
        lm = res.multi_face_landmarks[0].landmark

        h, w = depth.shape[:2]

        # 4) Collect 3D points with spatial median
        def collect_points(idxs):
            pts = []
            for i in idxs:
                u = int(lm[i].x * self.W)
                v = int(lm[i].y * self.H)
                ud = int(u * w / self.W)
                vd = int(v * h / self.H)
                ud = np.clip(ud, 0, w - 1)
                vd = np.clip(vd, 0, h - 1)
                Z = get_depth_median(depth, ud, vd, patch=1)
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                pts.append([X, Y, Z])
                cv2.circle(color, (u, v), 3, (0, 0, 255), -1)
            return np.array(pts)

        left_pts  = collect_points(self.left_idxs)
        right_pts = collect_points(self.right_idxs)

        # 5) Initial least‐squares fit
        left_c  = fit_sphere_center(left_pts)
        right_c = fit_sphere_center(right_pts)

        # Control: publish radius (should be 1.5 cm)
        def fit_sphere_radius(pts, center):
            dists = np.linalg.norm(pts - center[None, :], axis=1)
            r_mean = dists.mean()
            return r_mean
        
        left_r = fit_sphere_radius(left_pts, left_c)
        right_r = fit_sphere_radius(right_pts, right_c)

        self.get_logger().info(f"left_radius={left_r:.3f} m")
        self.get_logger().info(f"right_radius={right_r:.3f} m")

        # 6) Outlier rejection & refit
        def robust_fit(pts, c0):
            dists = np.linalg.norm(pts - c0[None,:], axis=1)
            mask  = np.abs(dists - self.R) < self.outlier_thresh
            if np.count_nonzero(mask) >= 4:
                return fit_sphere_center(pts[mask])
            return c0

        left_c  = robust_fit(left_pts, left_c)
        right_c = robust_fit(right_pts, right_c)

        # 7) Temporal smoothing (EMA) …
        if self.smoothed_left is None:
            self.smoothed_left  = left_c
            self.smoothed_right = right_c
        else:
            self.smoothed_left  = (
                self.alpha * left_c  + (1 - self.alpha) * self.smoothed_left
            )
            self.smoothed_right = (
                self.alpha * right_c + (1 - self.alpha) * self.smoothed_right
            )

        # 8) Transform into world frame 
        left_cam = self.smoothed_left 
        # flip Y so it’s “up” instead of “down”
        left_coppelia = np.array([ -left_cam[0],
                                   left_cam[1],
                                   -left_cam[2] ])
        
        right_cam = self.smoothed_right
        right_coppelia = np.array([ -right_cam[0],
                                   right_cam[1],
                                   -right_cam[2] ])

        #  R|T of camera to world:
        R = np.array([[-1,  0,  0],
                    [ 0,  0, -1],
                    [ 0, -1,  0]])
        R_toWorld = R.T
        T = np.array([-0.1301, -0.24275, 0.60139])   #Camera Position in the world

        # do your world‐frame
        left_world = R_toWorld.dot(left_coppelia) + T
        right_world = R_toWorld.dot(right_coppelia) + T

        # 9) Log both camera- and world- frame centers. Remember that left and right are switched up in the real world
        self.get_logger().info(
            f"[CAM]   Left center  = "
            f"(X={self.smoothed_left[0]:.3f}, "
            f"Y={self.smoothed_left[1]:.3f}, "
            f"Z={self.smoothed_left[2]:.3f}) m"
        )
        self.get_logger().info(
            f"[WORLD] Left center  = "
            f"(X={left_world[0]:.3f}, "
            f"Y={left_world[1]:.3f}, "
            f"Z={left_world[2]:.3f}) m"
        )
        self.get_logger().info(
            f"[CAM]   Right center = "
            f"(X={self.smoothed_right[0]:.3f}, "
            f"Y={self.smoothed_right[1]:.3f}, "
            f"Z={self.smoothed_right[2]:.3f}) m"
        )
        
        self.get_logger().info(
            f"[WORLD] Right center = "
            f"(X={right_world[0]:.3f}, "
            f"Y={right_world[1]:.3f}, "
            f"Z={right_world[2]:.3f}) m"
        )

        # 10) Display for sanity
        cv2.imshow('phase3', color)
        cv2.waitKey(1)


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