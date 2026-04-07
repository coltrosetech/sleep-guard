import cv2
import time
from utils.logger import setup_logger

logger = setup_logger()


class VideoSource:
    def __init__(self, config):
        self.config = config
        self.is_live = config.video_path is None

        if self.is_live:
            self._cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            logger.info(f"Kamera acildi: index={config.camera_index}")
        else:
            self._cap = cv2.VideoCapture(config.video_path)
            logger.info(f"Video dosyasi acildi: {config.video_path}")

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Video kaynagi acilamadi: "
                f"{'kamera ' + str(config.camera_index) if self.is_live else config.video_path}"
            )

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_delay = 1.0 / self.fps if not self.is_live else 0
        self._last_time = time.time()
        self._start_time = time.time()
        self._frame_count = 0

        logger.info(f"Cozunurluk: {self.width}x{self.height}, FPS: {self.fps:.1f}")

    def read(self):
        if not self.is_live and self._frame_delay > 0:
            elapsed = time.time() - self._last_time
            wait = self._frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

        ret, frame = self._cap.read()
        now = time.time()
        self._last_time = now
        timestamp = now - self._start_time

        if ret:
            self._frame_count += 1

        return ret, frame, timestamp

    @property
    def actual_fps(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed > 0 and self._frame_count > 0:
            return self._frame_count / elapsed
        return self.fps

    def release(self):
        if self._cap:
            self._cap.release()

    def __del__(self):
        self.release()
