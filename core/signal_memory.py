"""
Signal Memory - Son bilinen yuz/goz sinyallerini zamanla azalan agirlikla korur.

Problem: Yuz 1 frame kaybolunca tum goz verisi siliniyor.
Cozum: Son bilinen EAR, PERCLOS, bas pozu degerlerini 15sn half-life ile koru.
  - 5sn sonra: %79 guc
  - 15sn sonra: %50 guc
  - 30sn sonra: %25 guc
  - 60sn sonra: tamamen sil
"""
import math
from dataclasses import dataclass


@dataclass
class MemorizedSignals:
    ear: float = -1.0           # -1 = hic gozlemlenmedi
    ear_confidence: float = 0.0
    is_closed: bool = False
    perclos: float = 0.0
    head_pitch_dev: float = 0.0
    head_roll_dev: float = 0.0
    timestamp: float = 0.0

    @property
    def valid(self) -> bool:
        return self.ear >= 0

    def age(self, now: float) -> float:
        if self.timestamp <= 0:
            return float("inf")
        return now - self.timestamp


class SignalMemory:
    def __init__(self, half_life_sec: float = 15.0, max_sec: float = 60.0):
        self._signals = MemorizedSignals()
        self.half_life = half_life_sec
        self.max_sec = max_sec

    def update(self, eye_metrics, head_metrics, timestamp: float):
        """Taze yuz sinyallerini kaydet."""
        if eye_metrics and eye_metrics.ear_confidence > 0.3:
            self._signals.ear = eye_metrics.avg_ear
            self._signals.ear_confidence = eye_metrics.ear_confidence
            self._signals.is_closed = eye_metrics.is_closed
            self._signals.perclos = eye_metrics.perclos
            self._signals.timestamp = timestamp
        if head_metrics:
            self._signals.head_pitch_dev = head_metrics.pitch_deviation
            self._signals.head_roll_dev = head_metrics.roll_deviation
            if self._signals.timestamp < timestamp:
                self._signals.timestamp = timestamp

    def recall(self, now: float) -> tuple:
        """Hafizadaki sinyalleri ve azalma faktorunu don. (signals, decay_factor)"""
        age = self._signals.age(now)
        if age > self.max_sec or not self._signals.valid:
            return self._signals, 0.0
        factor = math.pow(0.5, age / self.half_life)
        return self._signals, factor

    def reset(self):
        self._signals = MemorizedSignals()
