import cv2
from config import GuardState

WINDOW_NAME = "SleepGuard v2 - Guvenlik Kabini Uyku Tespit"


class DisplayManager:
    def __init__(self, config):
        self.config = config
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    def show(self, frame, state: GuardState = GuardState.ACTIVE, fps: float = 0.0) -> str | None:
        h, w = frame.shape[:2]
        s = max(0.5, w / 1920)

        # FPS + resolution
        info = f"{fps:.0f} FPS | {w}x{h}"
        cv2.putText(frame, info, (w - int(170 * s), int(25 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.4 * s), (70, 70, 85), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            return "quit"
        elif key == ord('r'):
            return "recalibrate"
        elif key == ord('m'):
            self.config.show_mesh = not self.config.show_mesh
        elif key == ord('d'):
            self.config.show_debug = not self.config.show_debug
        elif key == ord('s'):
            self.config.show_skeleton = not self.config.show_skeleton
        elif key == ord('z'):
            self.config.show_zones = not self.config.show_zones
        elif key == ord('p'):
            while (cv2.waitKey(0) & 0xFF) not in (ord('p'), 27):
                pass
        return None

    def destroy(self):
        cv2.destroyAllWindows()
