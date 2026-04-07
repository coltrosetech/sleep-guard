import cv2
import mediapipe as mp
import numpy as np
import time
import os
from dataclasses import dataclass, field
from utils.logger import setup_logger

logger = setup_logger()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "pose_landmarker_lite.task"
)

# MediaPipe Pose 33 landmark indices
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

SKELETON_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


@dataclass
class PersonInfo:
    landmarks: list = None
    bbox: tuple = None          # (x1, y1, x2, y2)
    bbox_center: tuple = None
    bbox_area: float = 0.0
    estimated_distance: float = 0.0


@dataclass
class PersonResult:
    detected: bool = False
    landmarks: list | None = None
    bbox: tuple | None = None
    bbox_center: tuple | None = None
    bbox_area: float = 0.0
    estimated_distance: float = 0.0
    # Multi-person
    person_count: int = 0
    all_persons: list = field(default_factory=list)  # list of PersonInfo


class PersonDetector:
    def __init__(self, config):
        self.config = config

        if not os.path.exists(POSE_MODEL_PATH):
            logger.warning(f"Pose model bulunamadi: {POSE_MODEL_PATH}")
            self._download_model()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=5,  # Max 5 kisi tespit et
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._start_time = time.time()
        self._last_ts = 0
        logger.info("MediaPipe PoseLandmarker baslatildi (33 nokta, max 5 kisi)")

    def _download_model(self):
        import urllib.request
        os.makedirs(os.path.dirname(POSE_MODEL_PATH), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        logger.info("Pose model indiriliyor...")
        urllib.request.urlretrieve(url, POSE_MODEL_PATH)
        logger.info(f"Pose model indirildi: {os.path.getsize(POSE_MODEL_PATH)} bytes")

    def process(self, frame) -> PersonResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now_ms = int((time.time() - self._start_time) * 1000)
        if now_ms <= self._last_ts:
            now_ms = self._last_ts + 1
        self._last_ts = now_ms

        result = self._landmarker.detect_for_video(mp_image, now_ms)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return PersonResult(detected=False, person_count=0)

        # Tum kisileri isle
        all_persons = []
        for landmarks in result.pose_landmarks:
            info = self._extract_person(landmarks, w, h)
            if info:
                all_persons.append(info)

        if not all_persons:
            return PersonResult(detected=False, person_count=0)

        # En buyuk bbox = en yakin kisi = ana hedef
        primary = max(all_persons, key=lambda p: p.bbox_area)

        return PersonResult(
            detected=True,
            landmarks=primary.landmarks,
            bbox=primary.bbox,
            bbox_center=primary.bbox_center,
            bbox_area=primary.bbox_area,
            estimated_distance=primary.estimated_distance,
            person_count=len(all_persons),
            all_persons=all_persons,
        )

    def _extract_person(self, landmarks, w, h) -> PersonInfo | None:
        xs, ys = [], []
        for lm in landmarks:
            if lm.visibility > 0.3:
                xs.append(lm.x * w)
                ys.append(lm.y * h)

        if len(xs) < 3:
            return None

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = ((x2 - x1) * (y2 - y1)) / (w * h)
        bbox_height = y2 - y1
        dist = self._estimate_distance(bbox_height, h)

        return PersonInfo(
            landmarks=landmarks,
            bbox=(x1, y1, x2, y2),
            bbox_center=(cx, cy),
            bbox_area=area,
            estimated_distance=dist,
        )

    def _estimate_distance(self, bbox_height: int, frame_height: int) -> float:
        if bbox_height <= 0:
            return 5.0
        ratio = bbox_height / frame_height
        if ratio > 0.01:
            return 0.7 / ratio
        return 5.0

    def is_face_usable(self, landmarks, distance: float) -> bool:
        if landmarks is None or distance > self.config.face_distance_max:
            return False
        nose_vis = landmarks[NOSE].visibility
        l_eye_vis = landmarks[LEFT_EYE].visibility
        r_eye_vis = landmarks[RIGHT_EYE].visibility
        min_conf = self.config.face_keypoint_conf_min
        return nose_vis > min_conf and (l_eye_vis > min_conf or r_eye_vis > min_conf)

    def release(self):
        self._landmarker.close()
