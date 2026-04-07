"""
3 Modlu Sinyal Birlestirme (Fusion)

Modlar:
  full     - Yuz gorunuyor: canli goz + vucut sinyalleri
  memory   - Yuz az once gorunuyordu: hafizadaki goz + canli vucut
  body_only - Yuz uzun suredir yok: sadece vucut sinyalleri
"""
from dataclasses import dataclass, field
from config import SleepGuardConfig, CalibratedThresholds
from detectors.eye_tracker import EyeMetrics
from detectors.head_pose import HeadPoseMetrics
from detectors.movement_analyzer import MovementMetrics
from detectors.pose_analyzer import PoseMetrics
from core.zone_manager import ZoneStatus
from core.absence_tracker import AbsenceStatus
from core.signal_memory import SignalMemory
from utils.math_utils import clamp


@dataclass
class FusionResult:
    score: float = 0.0
    component_scores: dict = field(default_factory=dict)
    component_weights: dict = field(default_factory=dict)
    fusion_mode: str = "none"


class SignalFusion:
    def __init__(self, config: SleepGuardConfig):
        self.config = config

    def compute(
        self,
        eye: EyeMetrics | None,
        head: HeadPoseMetrics | None,
        movement: MovementMetrics | None,
        pose: PoseMetrics | None,
        face_visible: bool,
        face_lost_sec: float,
        thresholds: CalibratedThresholds,
        zone_status: ZoneStatus | None = None,
        absence_status: AbsenceStatus | None = None,
        signal_memory: SignalMemory | None = None,
        timestamp: float = 0.0,
    ) -> FusionResult:
        cfg = self.config

        if absence_status and absence_status.is_absent:
            return FusionResult(score=0.0, fusion_mode="absent")

        scores = {}
        weights = {}

        # ══════════════════════════════════════════════
        # MOD SECIMI
        # ══════════════════════════════════════════════
        if face_visible and eye and head:
            mode = "full"
            # Hafizayi taze veriyle guncelle
            if signal_memory:
                signal_memory.update(eye, head, timestamp)
        elif signal_memory:
            mem, decay = signal_memory.recall(timestamp)
            if decay > 0.05 and mem.valid:
                mode = "memory"
            else:
                mode = "body_only"
        else:
            mode = "body_only"

        # ══════════════════════════════════════════════
        # FULL MOD - Canli yuz + vucut
        # ══════════════════════════════════════════════
        if mode == "full":
            # Vucut durusu
            if pose and pose.pose_confidence > 0.3:
                pose_s = pose.pose_score
                if movement and pose.posture.value in ("SLOUCHED", "HEAD_DOWN"):
                    escalation = clamp(movement.stillness_duration_sec / 30.0) * 0.15
                    pose_s = min(pose_s + escalation, 1.0)
                scores["vucut"] = pose_s
                weights["vucut"] = cfg.w_body_pose_face

            # EAR (goz kapali ise agirlik artirilir)
            if eye.ear_confidence > 0.3:
                ear_range = max(thresholds.ear_baseline - thresholds.ear_closed, 1e-6)
                ear_score = 1.0 - clamp((eye.avg_ear - thresholds.ear_closed) / ear_range)
                scores["goz"] = ear_score
                ear_w = cfg.w_ear_face
                if eye.avg_ear < thresholds.ear_closed:
                    ear_w = min(ear_w * 1.5, 0.45)
                weights["goz"] = ear_w * eye.ear_confidence

            # PERCLOS
            if eye.ear_confidence > 0.3:
                scores["perclos"] = clamp(eye.perclos / 40.0)
                weights["perclos"] = cfg.w_perclos_face * eye.ear_confidence

            # Bas pozu
            pitch_s = clamp(head.pitch_deviation / max(cfg.head_pitch_sleeping, 1.0))
            roll_s = clamp(head.roll_deviation / max(cfg.head_roll_sleeping, 1.0))
            scores["bas"] = max(pitch_s, roll_s)
            weights["bas"] = cfg.w_head_pose_face

            # Vucut hareketi
            if movement:
                still_ratio = clamp(movement.stillness_duration_sec / max(cfg.movement_still_duration_sec, 1.0))
                eyes_closed = eye and eye.avg_ear < thresholds.ear_closed
                if pose and pose.posture.value == "UPRIGHT" and not eyes_closed:
                    still_ratio *= 0.3
                scores["hareket"] = still_ratio
                weights["hareket"] = cfg.w_body_movement_face

        # ══════════════════════════════════════════════
        # MEMORY MOD - Hafizadaki goz + canli vucut
        # ══════════════════════════════════════════════
        elif mode == "memory":
            mem, decay = signal_memory.recall(timestamp)

            # Vucut durusu (canli)
            if pose and pose.pose_confidence > 0.3:
                pose_s = pose.pose_score
                if movement and pose.posture.value in ("SLOUCHED", "HEAD_DOWN"):
                    escalation = clamp(movement.stillness_duration_sec / 30.0) * 0.25
                    pose_s = min(pose_s + escalation, 1.0)
                scores["vucut"] = pose_s
                weights["vucut"] = cfg.w_body_pose_face

            # EAR (hafizadan, decay ile azaltilmis)
            if mem.ear_confidence > 0.3:
                ear_range = max(thresholds.ear_baseline - thresholds.ear_closed, 1e-6)
                ear_score = 1.0 - clamp((mem.ear - thresholds.ear_closed) / ear_range)
                scores["goz_hafiza"] = ear_score
                ear_w = cfg.w_ear_face * decay
                if mem.is_closed:
                    ear_w = min(ear_w * 1.5, 0.45 * decay)
                weights["goz_hafiza"] = ear_w * mem.ear_confidence

            # PERCLOS (hafizadan)
            if mem.perclos > 0:
                scores["perclos_h"] = clamp(mem.perclos / 40.0)
                weights["perclos_h"] = cfg.w_perclos_face * decay * mem.ear_confidence

            # Bas pozu (hafizadan)
            if mem.head_pitch_dev > 0 or mem.head_roll_dev > 0:
                pitch_s = clamp(mem.head_pitch_dev / max(cfg.head_pitch_sleeping, 1.0))
                roll_s = clamp(mem.head_roll_dev / max(cfg.head_roll_sleeping, 1.0))
                scores["bas_h"] = max(pitch_s, roll_s)
                weights["bas_h"] = cfg.w_head_pose_face * decay

            # Vucut hareketi (canli - hafiza bazli goz durumuna gore)
            if movement:
                still_ratio = clamp(movement.stillness_duration_sec / max(cfg.movement_still_duration_sec, 1.0))
                if pose and pose.posture.value == "UPRIGHT" and not mem.is_closed:
                    still_ratio *= 0.3
                scores["hareket"] = still_ratio
                weights["hareket"] = cfg.w_body_movement_face

        # ══════════════════════════════════════════════
        # BODY_ONLY MOD - Sadece vucut sinyalleri
        # ══════════════════════════════════════════════
        else:
            # Vucut durusu
            if pose and pose.pose_confidence > 0.3:
                pose_s = pose.pose_score
                if movement and pose.posture.value in ("SLOUCHED", "HEAD_DOWN"):
                    escalation = clamp(movement.stillness_duration_sec / 30.0) * 0.25
                    pose_s = min(pose_s + escalation, 1.0)
                scores["vucut"] = pose_s
                weights["vucut"] = cfg.w_body_pose_noface

            # Vucut hareketi
            if movement:
                still_ratio = clamp(movement.stillness_duration_sec / max(cfg.movement_still_duration_sec, 1.0))
                if pose and pose.posture.value == "UPRIGHT":
                    still_ratio *= 0.3
                scores["hareket"] = still_ratio
                weights["hareket"] = cfg.w_body_movement_noface

            # Bolge baglami
            if zone_status and zone_status.current_zone:
                zone_bonus = cfg.zone_couch_bonus if zone_status.current_zone == "koltuk" else 0.0
                scores["bolge"] = clamp(zone_bonus)
                weights["bolge"] = cfg.w_zone_context_noface

            # Yuz kayip suresi
            if face_lost_sec > 1.0:
                scores["yuz_kayip"] = clamp(face_lost_sec / max(cfg.face_lost_sleeping_sec, 1.0))
                weights["yuz_kayip"] = cfg.w_face_vis_noface

        # Agirlikli ortalama
        total_w = sum(weights.values())
        if total_w > 0:
            final = sum(scores[k] * weights[k] for k in scores) / total_w
        else:
            final = 0.0

        return FusionResult(
            score=clamp(final),
            component_scores=scores,
            component_weights=weights,
            fusion_mode=mode,
        )
