import cv2
import numpy as np
from dataclasses import dataclass
from utils.math_utils import normalize_angle, ema_update


@dataclass
class HeadPoseMetrics:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    pitch_deviation: float = 0.0
    roll_deviation: float = 0.0


# MediaPipe landmark indices for head pose
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Generic 3D face model points (in cm, arbitrary scale)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -63.6, -12.5),       # Chin
    (-43.3, 32.7, -26.0),      # Left eye corner
    (43.3, 32.7, -26.0),       # Right eye corner
    (-28.9, -28.9, -24.1),     # Left mouth corner
    (28.9, -28.9, -24.1),      # Right mouth corner
], dtype=np.float64)


class HeadPoseEstimator:
    def __init__(self, config):
        self.config = config
        self._smooth_pitch = 0.0
        self._smooth_roll = 0.0
        self._smooth_yaw = 0.0
        self._smooth_alpha = 0.3
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def compute(self, landmarks, frame_shape) -> HeadPoseMetrics:
        if landmarks is None:
            return HeadPoseMetrics()

        h, w = frame_shape[:2]

        # Build camera matrix if needed
        if self._camera_matrix is None or self._camera_matrix[0, 2] != w / 2:
            focal = w
            self._camera_matrix = np.array([
                [focal, 0, w / 2],
                [0, focal, h / 2],
                [0, 0, 1],
            ], dtype=np.float64)

        # Extract 2D image points
        image_points = np.array([
            (landmarks[POSE_LANDMARKS["nose_tip"]].x * w,
             landmarks[POSE_LANDMARKS["nose_tip"]].y * h),
            (landmarks[POSE_LANDMARKS["chin"]].x * w,
             landmarks[POSE_LANDMARKS["chin"]].y * h),
            (landmarks[POSE_LANDMARKS["left_eye"]].x * w,
             landmarks[POSE_LANDMARKS["left_eye"]].y * h),
            (landmarks[POSE_LANDMARKS["right_eye"]].x * w,
             landmarks[POSE_LANDMARKS["right_eye"]].y * h),
            (landmarks[POSE_LANDMARKS["left_mouth"]].x * w,
             landmarks[POSE_LANDMARKS["left_mouth"]].y * h),
            (landmarks[POSE_LANDMARKS["right_mouth"]].x * w,
             landmarks[POSE_LANDMARKS["right_mouth"]].y * h),
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseMetrics()

        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles from rotation matrix directly
        # This gives more stable results than RQDecomp3x3
        import math
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))   # X rotation
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))            # Y rotation
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))    # Z rotation
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0.0

        pitch = normalize_angle(pitch)
        yaw = normalize_angle(yaw)
        roll = normalize_angle(roll)

        # Smooth with EMA
        self._smooth_pitch = ema_update(self._smooth_pitch, pitch, self._smooth_alpha)
        self._smooth_roll = ema_update(self._smooth_roll, roll, self._smooth_alpha)
        self._smooth_yaw = ema_update(self._smooth_yaw, yaw, self._smooth_alpha)

        return HeadPoseMetrics(
            pitch=self._smooth_pitch,
            roll=self._smooth_roll,
            yaw=self._smooth_yaw,
            pitch_deviation=0.0,  # Will be set by calibrator
            roll_deviation=0.0,
        )

    def set_baseline(self, baseline_pitch: float, baseline_roll: float, baseline_yaw: float = 0.0):
        """Initialize smooth values to baseline after calibration."""
        self._smooth_pitch = baseline_pitch
        self._smooth_roll = baseline_roll
        self._smooth_yaw = baseline_yaw

    def compute_deviations(self, metrics: HeadPoseMetrics, thresholds) -> HeadPoseMetrics:
        metrics.pitch_deviation = abs(metrics.pitch - thresholds.baseline_pitch)
        metrics.roll_deviation = abs(metrics.roll - thresholds.baseline_roll)
        return metrics
