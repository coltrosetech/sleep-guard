from config import GuardState, SleepGuardConfig
from core.absence_tracker import AbsenceStatus
from utils.logger import setup_logger

logger = setup_logger()

STATE_LABELS_TR = {
    GuardState.ACTIVE: "AKTIF",
    GuardState.IDLE: "HAREKETSIZ",
    GuardState.DROWSY: "UYUKLAMA",
    GuardState.SLEEPING: "UYUYOR",
    GuardState.ABSENT: "ALAN TERK",
}


class SleepStateMachine:
    def __init__(self, config: SleepGuardConfig):
        self.config = config
        self.state = GuardState.ACTIVE
        self.state_enter_time = 0.0
        self._pending_transition = None
        self._pending_since = 0.0

    def update(self, fusion_score: float, timestamp: float,
               person_detected: bool = True,
               absence_status: AbsenceStatus | None = None) -> GuardState:

        # ── ABSENT handling (highest priority) ──
        if absence_status and absence_status.is_absent:
            if self.state != GuardState.ABSENT:
                self._transition(GuardState.ABSENT, timestamp, 0.0)
            return self.state

        # ── Return from ABSENT ──
        if self.state == GuardState.ABSENT and person_detected:
            self._transition(GuardState.ACTIVE, timestamp, 0.0)
            return self.state

        # ── Score-based transitions ──
        target = self._score_to_target(fusion_score)

        if target != self.state:
            if target != self._pending_transition:
                self._pending_transition = target
                self._pending_since = timestamp
            else:
                elapsed = timestamp - self._pending_since
                required = self._get_duration(self.state, target)
                if elapsed >= required:
                    self._transition(target, timestamp, fusion_score)
        else:
            self._pending_transition = None

        return self.state

    def _transition(self, new_state: GuardState, timestamp: float, score: float):
        old = self.state
        self.state = new_state
        self.state_enter_time = timestamp
        self._pending_transition = None
        old_label = STATE_LABELS_TR.get(old, old.value)
        new_label = STATE_LABELS_TR.get(new_state, new_state.value)
        logger.info(f"Durum: {old_label} -> {new_label} (skor: {score:.2f})")

    def _score_to_target(self, score: float) -> GuardState:
        if score >= self.config.score_sleeping:
            return GuardState.SLEEPING
        elif score >= self.config.score_drowsy:
            return GuardState.DROWSY
        elif score >= self.config.score_idle:
            return GuardState.IDLE
        else:
            return GuardState.ACTIVE

    def _get_duration(self, from_state: GuardState, to_state: GuardState) -> float:
        if to_state == GuardState.ACTIVE:
            return self.config.active_recover_sec
        elif to_state == GuardState.IDLE:
            return self.config.idle_enter_sec
        elif to_state == GuardState.DROWSY:
            return self.config.drowsy_enter_sec
        elif to_state == GuardState.SLEEPING:
            return self.config.sleeping_enter_sec
        elif to_state == GuardState.ABSENT:
            return self.config.absent_enter_sec
        return 2.0

    def reset(self):
        self.state = GuardState.ACTIVE
        self.state_enter_time = 0.0
        self._pending_transition = None
