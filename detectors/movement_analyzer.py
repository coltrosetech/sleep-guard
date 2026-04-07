import cv2
import numpy as np
import time
from dataclasses import dataclass

# Body keypoints for movement tracking (shoulders, hips, nose)
BODY_TRACKING_INDICES = [0, 11, 12, 23, 24]


@dataclass
class MovementMetrics:
    displacement: float = 0.0
    avg_displacement: float = 0.0
    stillness_duration_sec: float = 0.0
    is_still: bool = False


class MovementAnalyzer:
    def __init__(self, config):
        self.config = config
        self._prev_body_positions = None
        self._prev_gray = None
        self._still_start = None
        self._displacement_history = []
        self._max_history = 90

    def compute_body(self, landmarks, frame, timestamp) -> MovementMetrics:
        """Body-level movement using pose landmarks."""
        displacement = 0.0

        if landmarks is not None:
            curr_positions = []
            for idx in BODY_TRACKING_INDICES:
                lm = landmarks[idx]
                if lm.visibility > 0.3:
                    curr_positions.append(np.array([lm.x, lm.y]))

            if len(curr_positions) >= 2:
                curr = np.array(curr_positions)
                if (self._prev_body_positions is not None
                        and len(self._prev_body_positions) == len(curr)):
                    diffs = np.linalg.norm(curr - self._prev_body_positions, axis=1)
                    displacement = float(np.mean(diffs))
                self._prev_body_positions = curr
            else:
                displacement = self._frame_diff(frame)
                self._prev_body_positions = None
        else:
            displacement = self._frame_diff(frame)
            self._prev_body_positions = None

        # History
        self._displacement_history.append(displacement)
        if len(self._displacement_history) > self._max_history:
            self._displacement_history.pop(0)

        avg_disp = float(np.mean(self._displacement_history)) if self._displacement_history else 0.0

        # Stillness
        is_still = displacement < self.config.body_movement_threshold
        if is_still:
            if self._still_start is None:
                self._still_start = timestamp
            stillness_sec = timestamp - self._still_start
        else:
            self._still_start = None
            stillness_sec = 0.0

        return MovementMetrics(
            displacement=displacement,
            avg_displacement=avg_disp,
            stillness_duration_sec=stillness_sec,
            is_still=is_still,
        )

    def _frame_diff(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            displacement = float(np.mean(diff) / 255.0)
        else:
            displacement = 0.0
        self._prev_gray = gray
        return displacement

    def reset(self):
        self._prev_body_positions = None
        self._prev_gray = None
        self._still_start = None
        self._displacement_history.clear()
