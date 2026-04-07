from dataclasses import dataclass, field
from enum import Enum


class GuardState(Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    DROWSY = "DROWSY"
    SLEEPING = "SLEEPING"
    ABSENT = "ABSENT"


class AlertLevel(Enum):
    NONE = "NONE"
    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    CRITICAL = "CRITICAL"
    ABSENCE = "ABSENCE"


class CalibrationState(Enum):
    CALIBRATING = "CALIBRATING"
    READY = "READY"


@dataclass
class CalibratedThresholds:
    # Face
    ear_baseline: float = 0.28
    ear_closed: float = 0.20
    ear_drowsy: float = 0.23
    baseline_pitch: float = 0.0
    baseline_roll: float = 0.0
    baseline_eye_width_left: float = 0.04
    baseline_eye_width_right: float = 0.04
    # Body
    baseline_torso_angle: float = 0.0
    baseline_head_drop: float = 0.0


@dataclass
class SleepGuardConfig:
    # ── Input ──
    camera_index: int = 0
    video_path: str | None = None

    # ── Calibration ──
    calibration_duration_sec: float = 30.0
    calibration_ear_percentile: float = 10.0
    calibration_min_frames: int = 20

    # ── EAR (face close-range) ──
    ear_closed_ratio: float = 0.72
    ear_drowsy_ratio: float = 0.82

    # ── PERCLOS ──
    perclos_window_sec: float = 60.0

    # ── Head pose (face close-range) ──
    head_pitch_drowsy: float = 20.0
    head_pitch_sleeping: float = 35.0
    head_roll_drowsy: float = 20.0
    head_roll_sleeping: float = 35.0

    # ── Body pose thresholds ──
    torso_upright_max: float = 15.0     # degrees - dik oturma limiti
    torso_slouch_max: float = 40.0      # degrees - cokmis oturma limiti
    torso_lying_min: float = 60.0       # degrees - yatma baslangici
    head_drop_threshold: float = 0.3    # normalized - bas dusme esigi
    arms_near_head_dist: float = 0.25   # normalized - koluna yaslanma esigi (genis)

    # ── Face usability ──
    face_distance_max: float = 1.5      # metre - bu otede yuz analizi atlanir
    face_keypoint_conf_min: float = 0.5  # min pose landmark visibility for face

    # ── Movement ──
    movement_still_threshold: float = 0.005
    movement_still_duration_sec: float = 10.0
    body_movement_threshold: float = 0.008

    # ── Idle detection ──
    idle_threshold_sec: float = 300.0   # 5 dakika hareketsizlik

    # ── Absence detection ──
    absence_alert_sec: float = 180.0    # 3 dakika kisi yok

    # ── Zone ──
    zones_file: str = "zones.json"
    zone_couch_bonus: float = 0.15

    # ── State machine ──
    score_idle: float = 0.15
    score_drowsy: float = 0.4
    score_sleeping: float = 0.6
    score_active: float = 0.3           # bu altinda ACTIVE'e don
    idle_enter_sec: float = 5.0
    drowsy_enter_sec: float = 2.0
    sleeping_enter_sec: float = 3.0
    active_recover_sec: float = 1.0
    absent_enter_sec: float = 3.0

    # ── Alerts ──
    alert_cooldown_sec: float = 5.0
    alert_escalation_sec: float = 15.0
    sound_enabled: bool = True

    # ── Fusion weights: face visible (close range) ──
    w_body_pose_face: float = 0.25
    w_ear_face: float = 0.25
    w_perclos_face: float = 0.15
    w_head_pose_face: float = 0.15
    w_body_movement_face: float = 0.15
    w_face_vis_face: float = 0.05

    # ── Fusion weights: face NOT visible (distance/turned) ──
    w_body_pose_noface: float = 0.40
    w_body_movement_noface: float = 0.35
    w_zone_context_noface: float = 0.15
    w_face_vis_noface: float = 0.10

    # ── Drift update ──
    drift_alpha: float = 0.01

    # ── EAR confidence ──
    eye_width_confidence_min: float = 0.60

    # ── Signal memory ──
    signal_memory_half_life: float = 15.0   # hafiza azalma yari omru (sn)
    signal_memory_max_sec: float = 60.0     # mutlak hafiza limiti
    face_hysteresis_sec: float = 1.5        # yuz titresim korumasi suresi

    # ── Face visibility ──
    face_lost_sleeping_sec: float = 20.0

    # ── UI ──
    show_mesh: bool = False
    show_debug: bool = True
    show_skeleton: bool = True
    show_zones: bool = True
