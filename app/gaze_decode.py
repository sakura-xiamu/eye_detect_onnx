"""把 L2CS 分箱 logits 换成画面坐标下的俯仰、偏航。"""

from __future__ import annotations

import numpy as np

BIN_COUNT = 90
BIN_WIDTH = 4.0
ANGLE_OFFSET = 180.0


def logits_to_degrees(logits: np.ndarray) -> float:
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.shape[0] != BIN_COUNT:
        raise ValueError(f"视线分箱数量应为 {BIN_COUNT}，实际是 {scores.shape[0]}")
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    probabilities = weights / np.sum(weights)
    indexes = np.arange(BIN_COUNT, dtype=np.float64)
    return float(np.sum(probabilities * indexes) * BIN_WIDTH - ANGLE_OFFSET)


def to_screen_angles(model_pitch: float, model_yaw: float) -> tuple[float, float]:
    """模型正俯仰朝画面上，模型正偏航朝画面左。页面约定正偏航朝画面右。"""
    return float(model_pitch), -float(model_yaw)
