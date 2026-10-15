"""人脸检测加视线估计。图片走单帧，摄像头走时间平滑。"""

from __future__ import annotations

import threading

import numpy as np

from app.geometry import GazeWindow, expand_xywh, limit_long_edge
from app.tracker import GazeTracker

FACE_EXPAND = 0.2
LONG_EDGE = 640


class GazeEngine:
    def __init__(self, detector, estimator, tracker: GazeTracker | None = None):
        self.detector = detector
        self.estimator = estimator
        self.tracker = tracker or GazeTracker()
        self._lock = threading.Lock()
        self._last_box: list[float] | None = None

    def reset_camera(self) -> None:
        with self._lock:
            self.tracker.reset()
            self._last_box = None

    def detect(self, image_bgr: np.ndarray, window: GazeWindow, temporal: bool) -> dict:
        with self._lock:
            pitch, yaw, box, width, height = self._infer(image_bgr)
            if temporal and pitch is None:
                judged = self.tracker.update(None, None, window)
                if judged["message"] == "未检测到人脸":
                    self._last_box = None
                    box = None
                else:
                    box = self._last_box
            elif temporal:
                judged = self.tracker.update(pitch, yaw, window)
                self._last_box = box
            elif pitch is None:
                judged = {
                    "pitch": None,
                    "yaw": None,
                    "looking_outside": False,
                    "directions": [],
                    "message": "未检测到人脸",
                }
            else:
                judged = self.tracker.classify_still(pitch, yaw, window)

            result = dict(judged)
            result["directions"] = list(judged["directions"])
            if result["pitch"] is not None:
                result["pitch"] = round(float(result["pitch"]), 2)
                result["yaw"] = round(float(result["yaw"]), 2)
            result["face_box"] = box
            result["image_width"] = width
            result["image_height"] = height
            return result

    def _infer(self, image_bgr: np.ndarray):
        height, width = image_bgr.shape[:2]
        working, scale = limit_long_edge(image_bgr, LONG_EDGE)
        faces = self.detector.detect(working)
        if len(faces) == 0:
            return None, None, None, width, height

        areas = faces[:, 2] * faces[:, 3]
        face = faces[int(np.argmax(areas))]
        x, y, box_w, box_h = [float(value) / scale for value in face[:4]]
        x1, y1, x2, y2 = expand_xywh(x, y, box_w, box_h, FACE_EXPAND, width, height)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None, None, None, width, height

        pitch, yaw = self.estimator.estimate(crop)
        box = [round(x, 1), round(y, 1), round(box_w, 1), round(box_h, 1)]
        return float(pitch), float(yaw), box, width, height
