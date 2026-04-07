import time
import os
import cv2
from config import SleepGuardConfig, CalibrationState, GuardState
from input.video_source import VideoSource
from detectors.person_detector import PersonDetector
from detectors.pose_analyzer import PoseAnalyzer
from detectors.face_detector import FaceDetector
from detectors.eye_tracker import EyeTracker
from detectors.head_pose import HeadPoseEstimator
from detectors.movement_analyzer import MovementAnalyzer
from core.calibrator import AdaptiveCalibrator
from core.fusion import SignalFusion, FusionResult
from core.state_machine import SleepStateMachine
from core.zone_manager import ZoneManager
from core.absence_tracker import AbsenceTracker
from core.signal_memory import SignalMemory
from alert.alert_manager import AlertManager
from ui.overlay import OverlayRenderer
from ui.display import DisplayManager
from utils.logger import setup_logger

logger = setup_logger()


class DetectionPipeline:
    def __init__(self, config: SleepGuardConfig):
        self.config = config
        logger.info("SleepGuard v2 baslatiliyor...")

        self.source = VideoSource(config)
        self.person_detector = PersonDetector(config)
        self.pose_analyzer = PoseAnalyzer(config)
        self.face_detector = FaceDetector(config)
        self.eye_tracker = EyeTracker(config)
        self.head_pose = HeadPoseEstimator(config)
        self.movement = MovementAnalyzer(config)
        self.calibrator = AdaptiveCalibrator(config)
        self.fusion = SignalFusion(config)
        self.state_machine = SleepStateMachine(config)
        self.signal_memory = SignalMemory(
            half_life_sec=config.signal_memory_half_life,
            max_sec=config.signal_memory_max_sec,
        )

        self.zone_manager = ZoneManager(config)
        if os.path.exists(config.zones_file):
            self.zone_manager.load_zones(config.zones_file)
        self.absence_tracker = AbsenceTracker(config)

        self.alert = AlertManager(config)
        self.overlay = OverlayRenderer(config)
        self.display = DisplayManager(config)

        # Face hysteresis state
        self._cached_eye = None
        self._cached_head = None
        self._face_lost_time = None

        self._fps_timer = time.time()
        self._fps_count = 0
        self._current_fps = 0.0
        logger.info("Tum bilesenler hazir. Tespit aninda basliyor!")

    def run(self):
        try:
            while True:
                ok, frame, timestamp = self.source.read()
                if not ok:
                    if not self.source.is_live:
                        logger.info("Video sona erdi.")
                    break

                # FPS
                self._fps_count += 1
                elapsed_fps = time.time() - self._fps_timer
                if elapsed_fps >= 1.0:
                    self._current_fps = self._fps_count / elapsed_fps
                    self._fps_count = 0
                    self._fps_timer = time.time()

                # ═══ ISIK IYILESTIRME (dusuk isik icin) ═══
                gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = gray_check.mean()
                if avg_brightness < 100:  # Dusuk isik tespit
                    # CLAHE uygula (her kanal icin)
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

                # ═══ LAYER 1: Person Detection ═══
                person = self.person_detector.process(frame)

                # ═══ Zone + Absence ═══
                zone_status = None
                if person.detected and person.bbox_center:
                    zone_status = self.zone_manager.check_occupancy(
                        person.bbox_center, timestamp
                    )
                absence_status = self.absence_tracker.update(
                    person.detected,
                    zone_status.current_zone if zone_status else None,
                    timestamp,
                )

                # ═══ LAYER 2: Body Pose ═══
                pose_metrics = None
                if person.detected and person.landmarks:
                    pose_metrics = self.pose_analyzer.compute(person.landmarks)

                # ═══ Body Movement ═══
                body_movement = self.movement.compute_body(
                    person.landmarks if person.detected else None,
                    frame, timestamp,
                )

                # ═══ LAYER 3: Face Analysis (HYSTERESIS ile) ═══
                eye_metrics = None
                head_metrics = None
                face_visible = False

                if (person.detected and
                        self.person_detector.is_face_usable(
                            person.landmarks, person.estimated_distance)):
                    face_result = self.face_detector.process(frame)
                    if face_result.detected:
                        # Yuz gorunuyor - canli veri
                        face_visible = True
                        self._face_lost_time = None
                        thresholds = self.calibrator.get_thresholds()
                        eye_metrics = self.eye_tracker.compute(
                            face_result.landmarks, thresholds
                        )
                        head_metrics = self.head_pose.compute(
                            face_result.landmarks, frame.shape
                        )
                        head_metrics = self.head_pose.compute_deviations(
                            head_metrics, thresholds
                        )
                        # Cache for hysteresis
                        self._cached_eye = eye_metrics
                        self._cached_head = head_metrics
                    else:
                        # Yuz bu frame'de kayip - hysteresis kontrol
                        now = time.time()
                        if self._face_lost_time is None:
                            self._face_lost_time = now
                        lost_dur = now - self._face_lost_time
                        if lost_dur < self.config.face_hysteresis_sec and self._cached_eye:
                            # Hysteresis penceresi icinde: cache kullan
                            face_visible = True
                            eye_metrics = self._cached_eye
                            head_metrics = self._cached_head

                # ═══ Arka Plan Kalibrasyon (ENGELLEMEZ) ═══
                if self.calibrator.is_refining and person.detected:
                    self.calibrator.feed(eye_metrics, head_metrics,
                                         body_movement, pose_metrics)
                    # Head pose baseline ayarla (ilk kez)
                    if not self.calibrator.is_refining:
                        t = self.calibrator.get_thresholds()
                        self.head_pose.set_baseline(t.baseline_pitch, t.baseline_roll)

                # ═══ Drift (sadece ACTIVE durumda) ═══
                if self.state_machine.state == GuardState.ACTIVE:
                    self.calibrator.drift_update(eye_metrics, head_metrics, pose_metrics)

                # ═══ Fusion (HER ZAMAN calisir) ═══
                fusion_result = self.fusion.compute(
                    eye_metrics, head_metrics, body_movement, pose_metrics,
                    face_visible, self.face_detector.face_lost_duration,
                    self.calibrator.get_thresholds(),
                    zone_status, absence_status,
                    signal_memory=self.signal_memory,
                    timestamp=timestamp,
                )

                # ═══ State Machine ═══
                state = self.state_machine.update(
                    fusion_result.score, timestamp,
                    person.detected, absence_status,
                )

                # ═══ Alerts ═══
                self.alert.update(state, timestamp, absence_status)

                # ═══ Render ═══
                # Tum kisilerin iskeletini ciz
                if person.detected and self.config.show_skeleton:
                    PERSON_COLORS = [(210,180,50),(80,220,120),(200,100,160),(60,210,255),(50,150,255)]
                    for idx, p in enumerate(person.all_persons):
                        color = PERSON_COLORS[idx % len(PERSON_COLORS)]
                        self.overlay.draw_skeleton(frame, p.landmarks, color)
                        # Bbox + kisi numarasi
                        if p.bbox:
                            x1, y1, x2, y2 = p.bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                            cv2.putText(frame, f"#{idx+1}", (x1, y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    # Kisi sayisi badge
                    if person.person_count > 1:
                        cv2.putText(frame, f"{person.person_count} kisi",
                                    (frame.shape[1]//2 - 40, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,220,120), 2, cv2.LINE_AA)
                if self.config.show_zones:
                    self.zone_manager.draw_zones(
                        frame, zone_status.current_zone if zone_status else None
                    )
                if not person.detected and absence_status.absence_duration > 10:
                    self.overlay.draw_absence_warning(frame, absence_status)
                elif not person.detected and absence_status.absence_duration > 1:
                    self.overlay.draw_no_person(frame, absence_status.absence_duration)

                # Kalibrasyon gostergesi (arka planda)
                if self.calibrator.is_refining:
                    self.overlay.draw_calibration_badge(frame, self.calibrator.progress)

                self.overlay.draw(
                    frame, state, fusion_result, eye_metrics, head_metrics,
                    body_movement, pose_metrics, zone_status, absence_status,
                    self.alert.current_level,
                )

                # ═══ Display ═══
                action = self.display.show(frame, state, self._current_fps)
                if action == "quit":
                    break
                elif action == "recalibrate":
                    self._recalibrate()
                elif action == "setup_zones":
                    ret, zf, _ = self.source.read()
                    if ret:
                        self.zone_manager.setup_interactive(zf)

        except KeyboardInterrupt:
            logger.info("Durduruldu.")
        finally:
            self._cleanup()

    def _recalibrate(self):
        self.calibrator.reset()
        self.state_machine.reset()
        self.movement.reset()
        self.pose_analyzer.reset()
        self.eye_tracker.reset()
        self.signal_memory.reset()
        logger.info("Yeniden kalibrasyon basladi.")

    def _cleanup(self):
        logger.info("Kapatiliyor...")
        self.source.release()
        self.person_detector.release()
        self.face_detector.release()
        self.display.destroy()
        logger.info("SleepGuard kapatildi.")
