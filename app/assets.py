"""准备 YuNet 与视线 ONNX。视线权重不提交。"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
FACE_FILE = WEIGHTS / "face_detection_yunet_2023mar.onnx"
GAZE_FILE = WEIGHTS / "mobileone_s0_gaze.onnx"

FACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
GAZE_URL = (
    "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"
    "mobileone_s0_gaze.onnx"
)


def ensure_models() -> None:
    _download(FACE_URL, FACE_FILE, min_bytes=100_000)
    _download(GAZE_URL, GAZE_FILE, min_bytes=1_000_000)


def _download(url: str, destination: Path, min_bytes: int) -> None:
    if destination.is_file() and destination.stat().st_size >= min_bytes:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "eye-gaze"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if temporary.stat().st_size < min_bytes:
        temporary.unlink(missing_ok=True)
        raise IOError(f"下载不完整: {destination.name}")
    temporary.replace(destination)
