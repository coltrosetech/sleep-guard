import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from utils.math_utils import clamp, ema_update

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


class PostureClass(Enum):
    UPRIGHT = "UPRIGHT"
    SLOUCHED = "SLOUCHED"
    HEAD_DOWN = "HEAD_DOWN"
    LYING = "LYING"
    UNKNOWN = "UNKNOWN"


@dataclass
class PoseMetrics:
    posture: PostureClass = PostureClass.UNKNOWN
    torso_angle: float = 0.0        # 0=dik, 90=yatay
    head_drop: float = 0.0          # 0=normal, pozitif=bas omuzun altinda
    body_horizontal: bool = False
    arms_near_head: bool = False
    pose_confidence: float = 0.0
    pose_score: float = 0.0         # 0.0=uyanik, 1.0=uyuyor


class PoseAnalyzer:
    def __init__(self, config):
        self.config = config
        self._smooth_torso = 0.0
        self._smooth_head_drop = 0.0
        self._smooth_alpha = 0.3

    def compute(self, landmarks) -> PoseMetrics:
        if landmarks is None:
            return PoseMetrics()

        # Check key landmark visibility
        key_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE]
        visibilities = [landmarks[i].visibility for i in key_indices]
        conf = sum(v for v in visibilities) / len(visibilities)

        if conf < 0.3:
            return PoseMetrics(pose_confidence=conf)

        # Extract key points
        l_shoulder = np.array([landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y])
        r_shoulder = np.array([landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y])
        l_hip = np.array([landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y])
        r_hip = np.array([landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y])
        nose = np.array([landmarks[NOSE].x, landmarks[NOSE].y])

        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2

        # ── Torso angle ──
        torso_vec = shoulder_mid - hip_mid  # from hip to shoulder
        vertical = np.array([0, -1])  # up in image coords (Y decreases upward)
        dot = np.dot(torso_vec, vertical)
        mag = np.linalg.norm(torso_vec) * np.linalg.norm(vertical)
        if mag > 1e-6:
            raw_torso = math.degrees(math.acos(clamp(dot / mag, -1.0, 1.0)))
        else:
            raw_torso = 0.0

        self._smooth_torso = ema_update(self._smooth_torso, raw_torso, self._smooth_alpha)
        torso_angle = self._smooth_torso

        # ── Head drop ──
        torso_length = np.linalg.norm(shoulder_mid - hip_mid)
        if torso_length > 1e-6:
            # Positive = head below shoulder line (in image coords Y increases down)
            raw_head_drop = (nose[1] - shoulder_mid[1]) / torso_length
        else:
            raw_head_drop = 0.0

        self._smooth_head_drop = ema_update(self._smooth_head_drop, raw_head_drop, self._smooth_alpha)
        head_drop = self._smooth_head_drop

        # ── Body horizontal ──
        y_diff = abs(shoulder_mid[1] - hip_mid[1])
        body_horizontal = y_diff < 0.05  # shoulders and hips at similar Y

        # ── Arms near head ──
        arms_near_head = False
        for wrist_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if landmarks[wrist_idx].visibility > 0.3:
                wrist = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y])
                dist_to_nose = np.linalg.norm(wrist - nose)
                if dist_to_nose < self.config.arms_near_head_dist:
                    arms_near_head = True
                    break

        # ── Classify posture ──
        posture = self._classify(torso_angle, head_drop, body_horizontal, arms_near_head)

        # ── Compute score ──
        pose_score = self._compute_score(posture, torso_angle, head_drop, arms_near_head)

        return PoseMetrics(
            posture=posture,
            torso_angle=torso_angle,
            head_drop=head_drop,
            body_horizontal=body_horizontal,
            arms_near_head=arms_near_head,
            pose_confidence=conf,
            pose_score=pose_score,
        )

    def _classify(self, torso_angle, head_drop, body_horizontal, arms_near_head) -> PostureClass:
        if body_horizontal:
            return PostureClass.LYING
        # Use head_drop as primary sleep indicator (more reliable than torso at varying camera angles)
        if head_drop > self.config.head_drop_threshold:
            return PostureClass.HEAD_DOWN
        if torso_angle > self.config.torso_slouch_max:
            return PostureClass.HEAD_DOWN
        if torso_angle > self.config.torso_upright_max or arms_near_head:
            return PostureClass.SLOUCHED
        return PostureClass.UPRIGHT

    def _compute_score(self, posture, torso_angle, head_drop, arms_near_head) -> float:
        score = 0.0

        if posture == PostureClass.UPRIGHT:
            score = clamp(torso_angle / self.config.torso_upright_max * 0.1, 0.0, 0.1)
        elif posture == PostureClass.SLOUCHED:
            t = (torso_angle - self.config.torso_upright_max) / \
                max(self.config.torso_slouch_max - self.config.torso_upright_max, 1.0)
            score = 0.3 + clamp(t) * 0.25  # 0.30-0.55
        elif posture == PostureClass.HEAD_DOWN:
            score = 0.55 + clamp(head_drop / 0.5) * 0.25  # 0.55-0.80
        elif posture == PostureClass.LYING:
            score = 0.80 + clamp(torso_angle / 90.0) * 0.20  # 0.80-1.0

        if arms_near_head:
            score = min(score + 0.12, 1.0)

        return clamp(score)

    def reset(self):
        self._smooth_torso = 0.0
        self._smooth_head_drop = 0.0
