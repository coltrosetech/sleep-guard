import cv2
import mediapipe as mp
import time
import os
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "face_landmarker.task")


@dataclass
class FaceResult:
    detected: bool
    landmarks: list | None = None
    confidence: float = 0.0


class FaceDetector:
    def __init__(self, config):
        self.config = config

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model dosyasi bulunamadi: {MODEL_PATH}\n"
                f"Lutfen assets/face_landmarker.task dosyasini indirin."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._start_time = time.time()
        self._frame_timestamp_ms = 0
        self._face_lost_start = None
        self.face_lost_duration = 0.0
        logger.info(f"MediaPipe FaceLandmarker baslatildi (478 landmark)")

    def process(self, frame) -> FaceResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonically increasing timestamp in ms
        now_ms = int((time.time() - self._start_time) * 1000)
        if now_ms <= self._frame_timestamp_ms:
            now_ms = self._frame_timestamp_ms + 1
        self._frame_timestamp_ms = now_ms
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            self._face_lost_start = None
            self.face_lost_duration = 0.0
            return FaceResult(detected=True, landmarks=landmarks, confidence=1.0)
        else:
            now = time.time()
            if self._face_lost_start is None:
                self._face_lost_start = now
            self.face_lost_duration = now - self._face_lost_start
            return FaceResult(detected=False)

    def draw_mesh(self, frame, landmarks):
        if landmarks is None:
            return
        h, w = frame.shape[:2]
        thickness = max(1, int(w / 960))
        dot_r = max(1, int(w / 640))

        # Eye contours
        left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        173, 157, 158, 159, 160, 161, 246]
        right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249,
                         263, 466, 388, 387, 386, 385, 384, 398]

        eye_color = (210, 180, 50)  # cyan
        for eye_idx in [left_eye_idx, right_eye_idx]:
            pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], eye_color, thickness, cv2.LINE_AA)

        # EAR landmark dots (the 6 key points per eye)
        ear_left = [33, 160, 158, 133, 153, 144]
        ear_right = [362, 385, 387, 263, 373, 380]
        dot_color = (80, 220, 120)  # green
        for idx_list in [ear_left, ear_right]:
            for i in idx_list:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                cv2.circle(frame, (x, y), dot_r + 1, dot_color, -1, cv2.LINE_AA)

        # Nose + chin + forehead reference dots
        for i in [1, 199, 10]:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(frame, (x, y), dot_r, (200, 100, 160), -1, cv2.LINE_AA)  # purple

        # Jawline (subtle)
        jaw_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                   361, 288, 397, 365, 379, 378, 400, 377, 152,
                   148, 176, 149, 150, 136, 172, 58, 132, 93,
                   234, 127, 162, 21, 54, 103, 67, 109, 10]
        jaw_color = (60, 60, 75)
        for j in range(len(jaw_idx) - 1):
            p1 = (int(landmarks[jaw_idx[j]].x * w), int(landmarks[jaw_idx[j]].y * h))
            p2 = (int(landmarks[jaw_idx[j + 1]].x * w), int(landmarks[jaw_idx[j + 1]].y * h))
            cv2.line(frame, p1, p2, jaw_color, 1, cv2.LINE_AA)

    def release(self):
        self._landmarker.close()
