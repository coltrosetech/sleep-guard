import time
from config import GuardState, AlertLevel, SleepGuardConfig
from alert.sound import SoundPlayer
from core.absence_tracker import AbsenceStatus
from utils.logger import setup_logger

logger = setup_logger()


class AlertManager:
    def __init__(self, config: SleepGuardConfig):
        self.config = config
        self.current_level = AlertLevel.NONE
        self._last_sound_time = 0.0
        self._sleeping_start = None
        self._sound = SoundPlayer() if config.sound_enabled else None

    def update(self, state: GuardState, timestamp: float,
               absence_status: AbsenceStatus | None = None):
        prev_level = self.current_level

        if state == GuardState.ACTIVE:
            self.current_level = AlertLevel.NONE
            self._sleeping_start = None
            return

        if state == GuardState.IDLE:
            self.current_level = AlertLevel.INFO

        elif state == GuardState.ABSENT:
            self.current_level = AlertLevel.ABSENCE

        elif state == GuardState.DROWSY:
            self.current_level = AlertLevel.WARNING
            self._sleeping_start = None

        elif state == GuardState.SLEEPING:
            if self._sleeping_start is None:
                self._sleeping_start = timestamp
            duration = timestamp - self._sleeping_start
            if duration > self.config.alert_escalation_sec:
                self.current_level = AlertLevel.CRITICAL
            else:
                self.current_level = AlertLevel.ALARM

        # Sound with cooldown
        if self.current_level not in (AlertLevel.NONE,) and self._sound:
            cooldown = {
                AlertLevel.INFO: 30.0,
                AlertLevel.WARNING: 5.0,
                AlertLevel.ALARM: 3.0,
                AlertLevel.CRITICAL: 2.0,
                AlertLevel.ABSENCE: 10.0,
            }.get(self.current_level, 5.0)

            if timestamp - self._last_sound_time >= cooldown:
                self._sound.play(self.current_level)
                self._last_sound_time = timestamp

        if prev_level != self.current_level and self.current_level != AlertLevel.NONE:
            logger.info(f"Alarm: {self.current_level.value}")
