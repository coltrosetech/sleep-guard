"""
Arka Plan Adaptif Kalibrasyon - Sistemi ENGELLEMEZ.

Sistem ilk frame'den itibaren varsayilan degerlerle calisir.
Arka planda veri toplar, yeterli veri gelince esikleri iyilestirir.
"""
import time
import numpy as np
from config import CalibrationState, CalibratedThresholds, SleepGuardConfig
from utils.logger import setup_logger
from utils.math_utils import ema_update

logger = setup_logger()


class AdaptiveCalibrator:
    def __init__(self, config: SleepGuardConfig):
        self.config = config
        # Sistem aninda READY - varsayilan degerlerle calisir
        self.state = CalibrationState.READY
        self._thresholds = CalibratedThresholds()  # ear_baseline=0.28, etc.

        # Arka plan kalibrasyon durumu
        self._bg_active = True
        self._ear_samples = []
        self._pitch_samples = []
        self._roll_samples = []
        self._eye_width_left_samples = []
        self._eye_width_right_samples = []
        self._torso_angle_samples = []
        self._head_drop_samples = []
        self._start_time = None
        self._has_face_data = False
        self._has_body_data = False
        self._refined = False

        logger.info("Kalibrasyon: varsayilan degerlerle basliyor (arka planda iyilestirilecek)")

    @property
    def progress(self) -> float:
        if self._refined or not self._bg_active:
            return 1.0
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(elapsed / self.config.calibration_duration_sec, 1.0)

    @property
    def is_refining(self) -> bool:
        return self._bg_active and not self._refined

    def get_thresholds(self) -> CalibratedThresholds:
        return self._thresholds

    def feed(self, eye_metrics=None, head_metrics=None, movement_metrics=None, pose_metrics=None):
        """Arka planda veri topla - sistemi ENGELLEMEZ."""
        if not self._bg_active:
            return

        if self._start_time is None:
            self._start_time = time.time()

        # Kamera isinma suresi - ilk 2 saniye atla
        elapsed = time.time() - self._start_time
        if elapsed < 2.0:
            return

        # Yuz verileri (EAR 0.10-0.50 arasi gecerli)
        if eye_metrics and 0.10 < eye_metrics.avg_ear < 0.50:
            self._ear_samples.append(eye_metrics.avg_ear)
            self._eye_width_left_samples.append(eye_metrics.left_eye_width)
            self._eye_width_right_samples.append(eye_metrics.right_eye_width)
            self._has_face_data = True

        if head_metrics:
            self._pitch_samples.append(head_metrics.pitch)
            self._roll_samples.append(head_metrics.roll)

        # Vucut verileri
        if pose_metrics and pose_metrics.pose_confidence > 0.3:
            self._torso_angle_samples.append(pose_metrics.torso_angle)
            self._head_drop_samples.append(pose_metrics.head_drop)
            self._has_body_data = True

        # Yeterli veri + sure doldu mu?
        if elapsed >= self.config.calibration_duration_sec:
            self._try_refine()

    def _try_refine(self):
        total = len(self._torso_angle_samples) + len(self._ear_samples)
        if total < self.config.calibration_min_frames:
            return  # Henuz yeterli veri yok, toplamaya devam

        # Yuz esikleri
        if self._has_face_data and len(self._ear_samples) > 10:
            ear_arr = np.array(self._ear_samples)
            mean_ear = np.mean(ear_arr)
            std_ear = np.std(ear_arr)
            if std_ear > 0:
                ear_filtered = ear_arr[np.abs(ear_arr - mean_ear) < 2 * std_ear]
            else:
                ear_filtered = ear_arr
            if len(ear_filtered) > 5:
                baseline_ear = float(np.percentile(ear_filtered, 25))
            else:
                baseline_ear = float(np.median(ear_arr))

            if baseline_ear >= 0.18:
                self._thresholds.ear_baseline = baseline_ear
                self._thresholds.ear_closed = baseline_ear * self.config.ear_closed_ratio
                self._thresholds.ear_drowsy = baseline_ear * self.config.ear_drowsy_ratio

        if self._pitch_samples:
            self._thresholds.baseline_pitch = float(np.median(self._pitch_samples))
            self._thresholds.baseline_roll = float(np.median(self._roll_samples))

        if self._eye_width_left_samples:
            self._thresholds.baseline_eye_width_left = float(np.median(self._eye_width_left_samples))
            self._thresholds.baseline_eye_width_right = float(np.median(self._eye_width_right_samples))

        # Vucut esikleri
        if self._has_body_data and len(self._torso_angle_samples) > 5:
            self._thresholds.baseline_torso_angle = float(np.median(self._torso_angle_samples))
            self._thresholds.baseline_head_drop = float(np.median(self._head_drop_samples))

        self._bg_active = False
        self._refined = True

        face_info = f"EAR={self._thresholds.ear_baseline:.3f}" if self._has_face_data else "yuz yok"
        body_info = f"govde={self._thresholds.baseline_torso_angle:.1f}" if self._has_body_data else "vucut yok"
        logger.info(f"Arka plan kalibrasyonu tamamlandi! {face_info}, {body_info}")

    def drift_update(self, eye_metrics=None, head_metrics=None, pose_metrics=None):
        if not self._refined:
            return
        alpha = self.config.drift_alpha

        if eye_metrics and eye_metrics.avg_ear > self._thresholds.ear_drowsy:
            self._thresholds.ear_baseline = ema_update(
                self._thresholds.ear_baseline, eye_metrics.avg_ear, alpha
            )
            self._thresholds.ear_closed = self._thresholds.ear_baseline * self.config.ear_closed_ratio
            self._thresholds.ear_drowsy = self._thresholds.ear_baseline * self.config.ear_drowsy_ratio

        if head_metrics:
            self._thresholds.baseline_pitch = ema_update(
                self._thresholds.baseline_pitch, head_metrics.pitch, alpha
            )
            self._thresholds.baseline_roll = ema_update(
                self._thresholds.baseline_roll, head_metrics.roll, alpha
            )

        if pose_metrics and pose_metrics.pose_confidence > 0.5:
            self._thresholds.baseline_torso_angle = ema_update(
                self._thresholds.baseline_torso_angle, pose_metrics.torso_angle, alpha
            )

    def reset(self):
        self._bg_active = True
        self._refined = False
        self._ear_samples.clear()
        self._pitch_samples.clear()
        self._roll_samples.clear()
        self._eye_width_left_samples.clear()
        self._eye_width_right_samples.clear()
        self._torso_angle_samples.clear()
        self._head_drop_samples.clear()
        self._start_time = None
        self._has_face_data = False
        self._has_body_data = False
        logger.info("Kalibrasyon sifirlandi - arka planda yeniden toplanacak")
