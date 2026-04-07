import numpy as np
from dataclasses import dataclass
from utils.math_utils import euclidean_distance
from utils.ring_buffer import RingBuffer


@dataclass
class EyeMetrics:
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    is_closed: bool = False
    perclos: float = 0.0
    ear_confidence: float = 1.0
    left_eye_width: float = 0.0
    right_eye_width: float = 0.0


# MediaPipe Face Mesh landmark indices for EAR
# Each eye: [P1(corner), P2(upper1), P3(upper2), P4(corner), P5(lower2), P6(lower1)]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Horizontal width landmarks (outer corners)
LEFT_EYE_WIDTH = [33, 133]
RIGHT_EYE_WIDTH = [362, 263]


class EyeTracker:
    def __init__(self, config):
        self.config = config
        max_frames = int(config.perclos_window_sec * 30)  # assume max 30fps
        self._perclos_buffer = RingBuffer(maxlen=max_frames)
        self._ear_smooth_buffer = RingBuffer(maxlen=5)

    def compute(self, landmarks, thresholds) -> EyeMetrics:
        if landmarks is None:
            return EyeMetrics()

        left_ear = self._compute_ear(landmarks, LEFT_EYE)
        right_ear = self._compute_ear(landmarks, RIGHT_EYE)

        # Clamp EAR to physically valid range (landmark noise can cause >0.5)
        left_ear = min(left_ear, 0.50)
        right_ear = min(right_ear, 0.50)
        avg_ear = (left_ear + right_ear) / 2.0

        # Smooth EAR with small buffer
        self._ear_smooth_buffer.append(avg_ear)
        smoothed_ear = self._ear_smooth_buffer.mean()

        # Compute eye widths for confidence
        left_width = self._compute_eye_width(landmarks, LEFT_EYE_WIDTH)
        right_width = self._compute_eye_width(landmarks, RIGHT_EYE_WIDTH)

        # EAR confidence based on eye width vs baseline
        confidence = self._compute_confidence(
            left_width, right_width, thresholds
        )

        # Determine if closed (using calibrated threshold)
        is_closed = smoothed_ear < thresholds.ear_closed

        # Update PERCLOS buffer
        self._perclos_buffer.append(1.0 if is_closed else 0.0)
        perclos = self._perclos_buffer.mean() * 100.0

        return EyeMetrics(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=smoothed_ear,
            is_closed=is_closed,
            perclos=perclos,
            ear_confidence=confidence,
            left_eye_width=left_width,
            right_eye_width=right_width,
        )

    def _compute_ear(self, landmarks, eye_indices) -> float:
        pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
        # Vertical distances
        v1 = euclidean_distance(pts[1], pts[5])  # P2-P6
        v2 = euclidean_distance(pts[2], pts[4])  # P3-P5
        # Horizontal distance
        h = euclidean_distance(pts[0], pts[3])    # P1-P4
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _compute_eye_width(self, landmarks, width_indices) -> float:
        p1 = (landmarks[width_indices[0]].x, landmarks[width_indices[0]].y)
        p2 = (landmarks[width_indices[1]].x, landmarks[width_indices[1]].y)
        return euclidean_distance(p1, p2)

    def _compute_confidence(self, left_w, right_w, thresholds) -> float:
        if thresholds.baseline_eye_width_left < 1e-6:
            return 1.0

        left_ratio = left_w / max(thresholds.baseline_eye_width_left, 1e-6)
        right_ratio = right_w / max(thresholds.baseline_eye_width_right, 1e-6)
        avg_ratio = (left_ratio + right_ratio) / 2.0

        min_ratio = self.config.eye_width_confidence_min
        if avg_ratio >= 1.0:
            return 1.0
        elif avg_ratio <= min_ratio:
            return 0.0
        else:
            return (avg_ratio - min_ratio) / (1.0 - min_ratio)

    def reset(self):
        self._perclos_buffer.clear()
        self._ear_smooth_buffer.clear()
