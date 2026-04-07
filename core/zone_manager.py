import cv2
import json
import os
import numpy as np
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger()


@dataclass
class ZoneConfig:
    name: str
    polygon: list  # list of (x, y) tuples
    color: tuple = (100, 100, 100)
    expected_activity: str = "work"  # "work", "rest", "transit"


@dataclass
class ZoneStatus:
    current_zone: str | None = None
    zone_enter_time: float = 0.0
    zone_duration: float = 0.0
    in_any_zone: bool = False


class ZoneManager:
    def __init__(self, config):
        self.config = config
        self.zones: list[ZoneConfig] = []
        self._current_zone: str | None = None
        self._zone_enter_time: float = 0.0

    def load_zones(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self.zones = []
            colors = [(0, 200, 0), (200, 200, 0), (200, 0, 0), (200, 0, 200)]
            for i, z in enumerate(data.get("zones", [])):
                self.zones.append(ZoneConfig(
                    name=z["name"],
                    polygon=[(p[0], p[1]) for p in z["polygon"]],
                    color=tuple(z.get("color", colors[i % len(colors)])),
                    expected_activity=z.get("activity", "work"),
                ))
            logger.info(f"{len(self.zones)} bolge yuklendi: {[z.name for z in self.zones]}")
            return True
        except Exception as e:
            logger.warning(f"Bolge dosyasi okunamadi: {e}")
            return False

    def save_zones(self, file_path: str):
        data = {"zones": []}
        for z in self.zones:
            data["zones"].append({
                "name": z.name,
                "polygon": z.polygon,
                "color": list(z.color),
                "activity": z.expected_activity,
            })
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Bolgeler kaydedildi: {file_path}")

    def check_occupancy(self, point: tuple, timestamp: float) -> ZoneStatus:
        if not self.zones or point is None:
            return ZoneStatus()

        px, py = point
        found_zone = None

        for zone in self.zones:
            contour = np.array(zone.polygon, dtype=np.int32)
            result = cv2.pointPolygonTest(contour, (float(px), float(py)), False)
            if result >= 0:
                found_zone = zone.name
                break

        if found_zone != self._current_zone:
            self._current_zone = found_zone
            self._zone_enter_time = timestamp
            if found_zone:
                logger.info(f"Bolge degisimi: {found_zone}")

        duration = timestamp - self._zone_enter_time if self._current_zone else 0.0

        return ZoneStatus(
            current_zone=self._current_zone,
            zone_enter_time=self._zone_enter_time,
            zone_duration=duration,
            in_any_zone=self._current_zone is not None,
        )

    def get_zone_weight(self, zone_name: str | None) -> float:
        if zone_name is None:
            return 0.0
        for z in self.zones:
            if z.name == zone_name and z.expected_activity == "rest":
                return self.config.zone_couch_bonus
        return 0.0

    def draw_zones(self, frame, current_zone: str | None):
        h, w = frame.shape[:2]
        for zone in self.zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            is_active = zone.name == current_zone

            # Semi-transparent fill for active zone
            if is_active:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], zone.color)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Border
            thickness = 2 if is_active else 1
            cv2.polylines(frame, [pts], True, zone.color, thickness, cv2.LINE_AA)

            # Label
            cx = int(np.mean([p[0] for p in zone.polygon]))
            cy = int(np.mean([p[1] for p in zone.polygon]))
            cv2.putText(frame, zone.name, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 1, cv2.LINE_AA)

    def setup_interactive(self, frame):
        """Interactive zone setup - click to define polygon vertices."""
        zones_data = []
        zone_names = ["masa", "koltuk", "kapi"]
        zone_activities = ["work", "rest", "transit"]
        colors = [(0, 200, 0), (0, 200, 200), (200, 0, 200)]

        for idx in range(3):
            points = []
            display = frame.copy()

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))

            win = f"Bolge Tanimla: {zone_names[idx]} (Sol tik=nokta, Enter=bitir, ESC=atla)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 720)
            cv2.setMouseCallback(win, mouse_callback)

            while True:
                show = display.copy()
                # Draw existing zones
                for z in zones_data:
                    pts = np.array(z["polygon"], dtype=np.int32)
                    cv2.polylines(show, [pts], True, tuple(z["color"]), 2)

                # Draw current points
                for i, p in enumerate(points):
                    cv2.circle(show, p, 5, colors[idx], -1)
                    if i > 0:
                        cv2.line(show, points[i - 1], p, colors[idx], 2)

                cv2.putText(show, f"Bolge: {zone_names[idx]} | Sol tik=nokta | ENTER=bitir | ESC=atla",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(win, show)
                key = cv2.waitKey(1) & 0xFF
                if key == 13 and len(points) >= 3:  # Enter
                    break
                elif key == 27:  # ESC - skip this zone
                    points = []
                    break

            cv2.destroyWindow(win)

            if len(points) >= 3:
                zones_data.append({
                    "name": zone_names[idx],
                    "polygon": points,
                    "color": list(colors[idx]),
                    "activity": zone_activities[idx],
                })

        # Save
        self.zones = []
        for z in zones_data:
            self.zones.append(ZoneConfig(
                name=z["name"],
                polygon=z["polygon"],
                color=tuple(z["color"]),
                expected_activity=z["activity"],
            ))

        if self.zones:
            self.save_zones(self.config.zones_file)

        return self.zones
