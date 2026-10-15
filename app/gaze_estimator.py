"""MobileOne-S0 视线估计。输出换成画面坐标的俯仰、偏航。"""

from __future__ import annotations

import numpy as np

from app.gaze_decode import logits_to_degrees, to_screen_angles

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_face(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1))[None])


class GazeEstimator:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        _, _, height, width = session.get_inputs()[0].shape
        self.input_size = (int(width), int(height))
        self.output_names = [item.name for item in session.get_outputs()]

    def estimate(self, crop_bgr: np.ndarray) -> tuple[float, float]:
        blob = preprocess_face(crop_bgr, self.input_size)
        values = self.session.run(self.output_names, {self.input_name: blob})
        yaw_logits, pitch_logits = _split_logits(self.output_names, values)
        model_pitch = logits_to_degrees(pitch_logits)
        model_yaw = logits_to_degrees(yaw_logits)
        return to_screen_angles(model_pitch, model_yaw)


def _split_logits(names: list[str], values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    by_name = {name.lower(): value for name, value in zip(names, values)}
    yaw = next((value for name, value in by_name.items() if "yaw" in name), None)
    pitch = next((value for name, value in by_name.items() if "pitch" in name), None)
    if yaw is None or pitch is None:
        if len(values) != 2:
            raise ValueError("视线模型需要俯仰和偏航两个输出")
        yaw, pitch = values[0], values[1]
    return yaw, pitch
