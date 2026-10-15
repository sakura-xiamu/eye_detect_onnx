"""视线窗口、出屏滞回，以及检测前的长边限制。"""

from __future__ import annotations

import numpy as np


class GazeWindow:
    def __init__(self, pitch_min: float, pitch_max: float, yaw_min: float, yaw_max: float):
        self.pitch_min = float(pitch_min)
        self.pitch_max = float(pitch_max)
        self.yaw_min = float(yaw_min)
        self.yaw_max = float(yaw_max)


def parse_range(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise ValueError("区间需要两个数，例如 -30,5")
    low = float(parts[0])
    high = float(parts[1])
    if low >= high:
        raise ValueError("区间左边必须小于右边")
    return low, high


def raw_outside(pitch: float, yaw: float, window: GazeWindow) -> bool:
    return (
        pitch < window.pitch_min
        or pitch > window.pitch_max
        or yaw < window.yaw_min
        or yaw > window.yaw_max
    )


def gaze_directions(pitch: float, yaw: float, window: GazeWindow) -> list[str]:
    directions: list[str] = []
    if pitch > window.pitch_max:
        directions.append("上")
    if pitch < window.pitch_min:
        directions.append("下")
    if yaw < window.yaw_min:
        directions.append("左")
    if yaw > window.yaw_max:
        directions.append("右")
    return directions


def next_outside(
    outside: bool,
    pitch: float,
    yaw: float,
    window: GazeWindow,
    margin: float = 3.0,
) -> bool:
    if not outside:
        return (
            pitch < window.pitch_min - margin
            or pitch > window.pitch_max + margin
            or yaw < window.yaw_min - margin
            or yaw > window.yaw_max + margin
        )
    inside = (
        window.pitch_min + margin <= pitch <= window.pitch_max - margin
        and window.yaw_min + margin <= yaw <= window.yaw_max - margin
    )
    return not inside


def expand_xywh(
    x: float,
    y: float,
    width: float,
    height: float,
    factor: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """每边向外扩 factor 倍宽高，并裁到图像内。返回 x1, y1, x2, y2。"""
    x1 = max(0, int(np.floor(x - width * factor)))
    y1 = max(0, int(np.floor(y - height * factor)))
    x2 = min(image_width, int(np.ceil(x + width + width * factor)))
    y2 = min(image_height, int(np.ceil(y + height + height * factor)))
    return x1, y1, x2, y2


def limit_long_edge(image: np.ndarray, long_edge: int = 640) -> tuple[np.ndarray, float]:
    """把最长边压到 long_edge。返回缩放后的图，以及相对原图的比例。"""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_edge:
        return image, 1.0
    scale = long_edge / float(longest)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    import cv2

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale
