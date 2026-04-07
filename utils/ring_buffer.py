from collections import deque


class RingBuffer:
    def __init__(self, maxlen: int):
        self._buffer = deque(maxlen=maxlen)

    def append(self, value):
        self._buffer.append(value)

    def clear(self):
        self._buffer.clear()

    @property
    def data(self) -> list:
        return list(self._buffer)

    def __len__(self):
        return len(self._buffer)

    @property
    def full(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def mean(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    def ratio_above(self, threshold: float) -> float:
        if not self._buffer:
            return 0.0
        return sum(1 for v in self._buffer if v > threshold) / len(self._buffer)

    def ratio_below(self, threshold: float) -> float:
        if not self._buffer:
            return 0.0
        return sum(1 for v in self._buffer if v < threshold) / len(self._buffer)
