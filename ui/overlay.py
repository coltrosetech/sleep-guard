import cv2
import numpy as np
import time
from config import GuardState, AlertLevel
from detectors.person_detector import SKELETON_CONNECTIONS

# ─── Colors (BGR) ───
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
NEAR_BLACK = (20, 20, 25)
DARK_GRAY = (40, 40, 50)
MID_GRAY = (100, 100, 115)
LIGHT_GRAY = (180, 180, 195)
GREEN = (80, 220, 120)
YELLOW = (60, 210, 255)
RED = (70, 70, 255)
CYAN = (210, 180, 50)
PURPLE = (200, 100, 160)
ORANGE = (50, 150, 255)

STATE_COLORS = {
    GuardState.ACTIVE: GREEN,
    GuardState.IDLE: CYAN,
    GuardState.DROWSY: YELLOW,
    GuardState.SLEEPING: RED,
    GuardState.ABSENT: PURPLE,
}

STATE_LABELS = {
    GuardState.ACTIVE: "AKTIF",
    GuardState.IDLE: "HAREKETSIZ",
    GuardState.DROWSY: "UYUKLAMA",
    GuardState.SLEEPING: "UYUYOR!",
    GuardState.ABSENT: "ALAN TERK!",
}


def rounded_rect(img, pt1, pt2, color, radius, thickness=-1, alpha=1.0):
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if alpha < 1.0:
        roi = img[y1:y2, x1:x2].copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        region = img[y1:y2, x1:x2]
        cv2.addWeighted(region, alpha, roi, 1 - alpha, 0, region)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def bar(img, x, y, w, h, progress, fg, bg=DARK_GRAY):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fw = max(1, int(w * progress))
    cv2.rectangle(img, (x, y), (x + fw, y + h), fg, -1)


def txt(img, text, pos, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class OverlayRenderer:
    def __init__(self, config):
        self.config = config
        self._flash = 0

    # ─── CALIBRATION ───
    def draw_calibration(self, frame, progress: float):
        h, w = frame.shape[:2]
        s = max(0.5, w / 1920)

        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)

        # Card
        cw, ch = int(500 * s), int(200 * s)
        cx, cy = (w - cw) // 2, (h - ch) // 2
        rounded_rect(frame, (cx, cy), (cx + cw, cy + ch), NEAR_BLACK, 10, alpha=0.85)

        # Title
        txt(frame, "KALIBRASYON", (cx + int(140 * s), cy + int(50 * s)), 0.9 * s, WHITE, 2)
        txt(frame, "Normal pozisyonunuzda oturun...",
            (cx + int(80 * s), cy + int(80 * s)), 0.45 * s, MID_GRAY, 1)

        # Progress bar
        bx = cx + int(50 * s)
        by = cy + int(110 * s)
        bw = cw - int(100 * s)
        bar(frame, bx, by, bw, int(10 * s), progress, GREEN)

        # Percentage
        pct = f"{int(progress * 100)}%"
        remaining = max(0, self.config.calibration_duration_sec * (1 - progress))
        txt(frame, pct, (bx, by + int(30 * s)), 0.5 * s, GREEN, 1)
        txt(frame, f"{remaining:.0f}s", (bx + bw - int(40 * s), by + int(30 * s)), 0.5 * s, MID_GRAY, 1)

        txt(frame, "R:Yeniden  Q:Cikis",
            (cx + int(140 * s), cy + ch - int(20 * s)), 0.4 * s, DARK_GRAY, 1)

    # ─── CALIBRATION BADGE (small, non-blocking) ───
    def draw_calibration_badge(self, frame, progress: float):
        h, w = frame.shape[:2]
        s = max(0.5, w / 1920)
        bw = int(160 * s)
        bh = int(24 * s)
        bx = w - bw - int(14 * s)
        by = int(50 * s)
        rounded_rect(frame, (bx, by), (bx + bw, by + bh), NEAR_BLACK, 6, alpha=0.7)
        bar(frame, bx + 4, by + 4, bw - 8, bh - 8, progress, CYAN, DARK_GRAY)
        txt(frame, f"Kalibre {int(progress*100)}%", (bx + 8, by + bh - int(6 * s)),
            max(0.28, 0.32 * s), CYAN, 1)

    # ─── SKELETON ───
    def draw_skeleton(self, frame, landmarks, color=CYAN):
        if landmarks is None:
            return
        h, w = frame.shape[:2]
        thick = max(1, int(w / 640))
        dot_r = max(2, int(w / 480))

        # Bones
        for i1, i2 in SKELETON_CONNECTIONS:
            lm1, lm2 = landmarks[i1], landmarks[i2]
            if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                p1 = (int(lm1.x * w), int(lm1.y * h))
                p2 = (int(lm2.x * w), int(lm2.y * h))
                cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)

        # Joints
        for lm in landmarks:
            if lm.visibility > 0.3:
                p = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, p, dot_r, GREEN, -1, cv2.LINE_AA)

    # ─── MAIN HUD ───
    def draw(self, frame, state, fusion, eye, head, movement, pose,
             zone_status, absence_status, alert_level):
        h, w = frame.shape[:2]
        s = max(0.5, w / 1920)
        color = STATE_COLORS.get(state, GREEN)
        label = STATE_LABELS.get(state, "?")

        # Flash effect
        if alert_level == AlertLevel.CRITICAL:
            self._flash += 1
            if self._flash % 8 < 4:
                ov = np.zeros_like(frame)
                ov[:] = (0, 0, 40)
                cv2.addWeighted(frame, 0.85, ov, 0.15, 0, frame)

        # Border
        border_t = max(2, int(3 * s))
        if alert_level in (AlertLevel.ALARM, AlertLevel.CRITICAL, AlertLevel.ABSENCE):
            border_t = max(4, int(5 * s))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, border_t)

        # ── Status bar (top-left) ──
        bar_h = max(44, int(50 * s))
        bar_w = max(280, int(340 * s))
        m = max(10, int(14 * s))
        rounded_rect(frame, (m, m), (m + bar_w, m + bar_h), NEAR_BLACK, 10, alpha=0.8)

        # Dot
        dx = m + int(20 * s)
        dy = m + bar_h // 2
        cv2.circle(frame, (dx, dy), max(5, int(7 * s)), color, -1, cv2.LINE_AA)
        if state not in (GuardState.ACTIVE,):
            pulse = abs(np.sin(time.time() * 4)) * 0.7 + 0.3
            cv2.circle(frame, (dx, dy), int((10 + 5 * pulse) * s), color, 1, cv2.LINE_AA)

        # Label
        txt(frame, label, (dx + int(18 * s), m + int(bar_h * 0.65)),
            max(0.6, 0.7 * s), color, max(1, int(2 * s)))

        # Score
        score_text = f"{fusion.score:.0%}"
        sc = GREEN if fusion.score < 0.4 else (YELLOW if fusion.score < 0.7 else RED)
        txt(frame, score_text, (m + bar_w - int(65 * s), m + int(bar_h * 0.65)),
            max(0.55, 0.6 * s), sc, 1)

        # Mode badge
        mode = fusion.fusion_mode
        mode_label = {"full": "YUZ+VUCUT", "body_only": "VUCUT", "absent": "YOK", "none": "-"}.get(mode, mode)
        txt(frame, mode_label, (m + bar_w - int(65 * s), m + int(18 * s)),
            max(0.3, 0.35 * s), MID_GRAY, 1)

        # ── Debug panel ──
        if self.config.show_debug:
            self._draw_debug(frame, eye, head, movement, pose, fusion, zone_status, s)

        # ── Keyboard hints ──
        hints = "Q:Cikis R:Kalibre M:Mesh D:Debug S:Iskelet Z:Bolge"
        txt(frame, hints, (int(10 * s), h - int(10 * s)), max(0.3, 0.35 * s), DARK_GRAY, 1)

    # ─── DEBUG PANEL ───
    def _draw_debug(self, frame, eye, head, movement, pose, fusion, zone_status, s):
        h, w = frame.shape[:2]
        m = max(10, int(14 * s))
        pw = max(260, int(310 * s))
        ph = max(250, int(300 * s))
        px, py = m, h - ph - m

        rounded_rect(frame, (px, py), (px + pw, py + ph), NEAR_BLACK, 10, alpha=0.8)

        y = py + int(18 * s)
        lx = px + int(10 * s)
        vx = px + int(110 * s)
        bx = px + int(170 * s)
        bw = pw - int(185 * s)
        bh = max(5, int(6 * s))
        fs = max(0.35, 0.4 * s)
        rh = max(20, int(24 * s))

        txt(frame, "METRIKLER", (lx, y), max(0.32, 0.36 * s), MID_GRAY, 1)
        y += rh

        # Pose
        if pose and pose.pose_confidence > 0:
            pc = GREEN if pose.torso_angle < 15 else (YELLOW if pose.torso_angle < 40 else RED)
            txt(frame, "Govde", (lx, y), fs, LIGHT_GRAY, 1)
            txt(frame, f"{pose.torso_angle:.0f} {pose.posture.value[:4]}", (vx, y), fs, pc, 1)
            bar(frame, bx, y - bh - 1, bw, bh, min(pose.torso_angle / 60, 1), pc)
            y += rh

            txt(frame, "Bas Dusme", (lx, y), fs, LIGHT_GRAY, 1)
            hdc = GREEN if pose.head_drop < 0.15 else (YELLOW if pose.head_drop < 0.3 else RED)
            txt(frame, f"{pose.head_drop:.2f}", (vx, y), fs, hdc, 1)
            bar(frame, bx, y - bh - 1, bw, bh, min(abs(pose.head_drop) / 0.5, 1), hdc)
            y += rh

        # EAR
        if eye:
            ec = GREEN if eye.avg_ear > 0.22 else (YELLOW if eye.avg_ear > 0.16 else RED)
            txt(frame, "EAR", (lx, y), fs, LIGHT_GRAY, 1)
            txt(frame, f"{eye.avg_ear:.3f}", (vx, y), fs, ec, 1)
            bar(frame, bx, y - bh - 1, bw, bh, min(eye.avg_ear / 0.35, 1), ec)
            y += rh

            txt(frame, "PERCLOS", (lx, y), fs, LIGHT_GRAY, 1)
            plc = GREEN if eye.perclos < 15 else (YELLOW if eye.perclos < 30 else RED)
            txt(frame, f"{eye.perclos:.1f}%", (vx, y), fs, plc, 1)
            bar(frame, bx, y - bh - 1, bw, bh, min(eye.perclos / 50, 1), plc)
            y += rh

        # Movement
        if movement:
            if movement.is_still:
                mc = RED if movement.stillness_duration_sec > 60 else (YELLOW if movement.stillness_duration_sec > 10 else GREEN)
                mt = f"Sabit {movement.stillness_duration_sec:.0f}s"
            else:
                mc, mt = GREEN, "Aktif"
            txt(frame, "Hareket", (lx, y), fs, LIGHT_GRAY, 1)
            txt(frame, mt, (vx, y), fs, mc, 1)
            y += rh

        # Zone
        if zone_status and zone_status.current_zone:
            txt(frame, "Bolge", (lx, y), fs, LIGHT_GRAY, 1)
            txt(frame, f"{zone_status.current_zone} ({zone_status.zone_duration:.0f}s)",
                (vx, y), fs, CYAN, 1)
            y += rh

        # Fusion signals
        y += int(5 * s)
        txt(frame, "SINYALLER", (lx, y), max(0.3, 0.33 * s), MID_GRAY, 1)
        y += int(16 * s)
        for key, score in fusion.component_scores.items():
            sc = GREEN if score < 0.4 else (YELLOW if score < 0.7 else RED)
            txt(frame, key[:6], (lx, y), max(0.28, 0.32 * s), MID_GRAY, 1)
            bar(frame, lx + int(50 * s), y - bh - 1, pw - int(75 * s), bh, score, sc)
            y += int(14 * s)

    # ─── ABSENCE WARNING ───
    def draw_absence_warning(self, frame, absence_status):
        h, w = frame.shape[:2]
        s = max(0.5, w / 1920)
        dur = absence_status.absence_duration

        if absence_status.is_absent:
            # Full screen warning
            ov = np.zeros_like(frame)
            ov[:] = (40, 0, 40)
            pulse = abs(np.sin(time.time() * 2)) * 0.15 + 0.1
            cv2.addWeighted(frame, 1 - pulse, ov, pulse, 0, frame)

            text = f"ALAN TERK EDILDI!  {dur:.0f}s"
            ts = max(0.8, 1.2 * s)
            sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts, 3)[0]
            tx = (w - sz[0]) // 2
            txt(frame, text, (tx, h // 2), ts, PURPLE, 3)

            if absence_status.last_seen_zone:
                sub = f"Son bolge: {absence_status.last_seen_zone}"
                txt(frame, sub, (tx + int(50 * s), h // 2 + int(40 * s)), 0.5 * s, MID_GRAY, 1)

    # ─── NO PERSON ───
    def draw_no_person(self, frame, duration: float):
        h, w = frame.shape[:2]
        s = max(0.5, w / 1920)
        text = f"Kisi tespit edilemiyor ({duration:.0f}s)"
        sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, 2)[0]
        tx = (w - sz[0]) // 2
        ty = int(80 * s)
        rounded_rect(frame, (tx - 15, ty - sz[1] - 10), (tx + sz[0] + 15, ty + 10), NEAR_BLACK, 8, alpha=0.8)
        txt(frame, text, (tx, ty), 0.55 * s, ORANGE, 2)
