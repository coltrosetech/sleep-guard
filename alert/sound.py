import threading
import winsound
from config import AlertLevel


class SoundPlayer:
    def __init__(self):
        self._playing = False

    def play(self, level: AlertLevel):
        if self._playing:
            return
        thread = threading.Thread(target=self._play_sound, args=(level,), daemon=True)
        thread.start()

    def _play_sound(self, level: AlertLevel):
        self._playing = True
        try:
            if level == AlertLevel.INFO:
                winsound.Beep(800, 150)
            elif level == AlertLevel.WARNING:
                winsound.Beep(1000, 200)
            elif level == AlertLevel.ALARM:
                winsound.Beep(2500, 500)
            elif level == AlertLevel.CRITICAL:
                for _ in range(3):
                    winsound.Beep(2500, 150)
                    winsound.Beep(1500, 150)
            elif level == AlertLevel.ABSENCE:
                for _ in range(3):
                    winsound.Beep(600, 300)
        except Exception:
            pass
        finally:
            self._playing = False
