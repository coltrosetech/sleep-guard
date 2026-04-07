import time
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger()


@dataclass
class AbsenceStatus:
    is_absent: bool = False
    absence_duration: float = 0.0
    last_seen_zone: str | None = None
    last_seen_time: float = 0.0


class AbsenceTracker:
    def __init__(self, config):
        self.config = config
        self._last_person_time = time.time()
        self._last_zone: str | None = None
        self._alerted = False

    def update(self, person_detected: bool, zone: str | None, timestamp: float) -> AbsenceStatus:
        if person_detected:
            if self._alerted:
                logger.info("Kisi geri dondu.")
                self._alerted = False
            self._last_person_time = timestamp
            self._last_zone = zone
            return AbsenceStatus(
                is_absent=False,
                absence_duration=0.0,
                last_seen_zone=zone,
                last_seen_time=timestamp,
            )

        duration = timestamp - self._last_person_time
        is_absent = duration >= self.config.absence_alert_sec

        if is_absent and not self._alerted:
            logger.info(f"ALAN TERK: {duration:.0f}s kisi yok! Son bolge: {self._last_zone}")
            self._alerted = True

        return AbsenceStatus(
            is_absent=is_absent,
            absence_duration=duration,
            last_seen_zone=self._last_zone,
            last_seen_time=self._last_person_time,
        )

    def reset(self):
        self._last_person_time = time.time()
        self._last_zone = None
        self._alerted = False
