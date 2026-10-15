"""摄像头上的角度中值、出屏滞回，以及漏检保持。"""

from __future__ import annotations

from collections import deque

import numpy as np

from app.geometry import GazeWindow, gaze_directions, next_outside, raw_outside


class GazeTracker:
    def __init__(self, history: int = 5, margin: float = 3.0, miss_limit: int = 10):
        self.history = history
        self.margin = margin
        self.miss_limit = miss_limit
        self._pitches: deque[float] = deque(maxlen=history)
        self._yaws: deque[float] = deque(maxlen=history)
        self._outside = False
        self._misses = 0
        self._last: dict | None = None

    def reset(self) -> None:
        self._pitches.clear()
        self._yaws.clear()
        self._outside = False
        self._misses = 0
        self._last = None

    def classify_still(self, pitch: float, yaw: float, window: GazeWindow) -> dict:
        outside = raw_outside(pitch, yaw, window)
        return self._result(pitch, yaw, outside, window)

    def update(
        self,
        pitch: float | None,
        yaw: float | None,
        window: GazeWindow,
    ) -> dict:
        if pitch is None or yaw is None:
            return self._miss()
        self._misses = 0
        self._pitches.append(float(pitch))
        self._yaws.append(float(yaw))
        smooth_pitch = float(np.median(self._pitches))
        smooth_yaw = float(np.median(self._yaws))
        self._outside = next_outside(
            self._outside,
            smooth_pitch,
            smooth_yaw,
            window,
            self.margin,
        )
        result = self._result(smooth_pitch, smooth_yaw, self._outside, window)
        self._last = result
        return result

    def _miss(self) -> dict:
        if self._last is None:
            return self._empty()
        self._misses += 1
        if self._misses >= self.miss_limit:
            self.reset()
            return self._empty()
        return dict(self._last)

    def _result(self, pitch: float, yaw: float, outside: bool, window: GazeWindow) -> dict:
        directions = gaze_directions(pitch, yaw, window) if outside else []
        return {
            "pitch": pitch,
            "yaw": yaw,
            "looking_outside": outside,
            "directions": directions,
            "message": "屏幕外" if outside else "屏幕内",
        }

    def _empty(self) -> dict:
        return {
            "pitch": None,
            "yaw": None,
            "looking_outside": False,
            "directions": [],
            "message": "未检测到人脸",
        }
