"""加载两个 ONNX，确认 DirectML 能跑通一帧空白图和一块人脸裁剪。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.assets import FACE_FILE, GAZE_FILE
from app.engine import GazeEngine
from app.face_detector import YuNetDetector
from app.gaze_estimator import GazeEstimator
from app.geometry import GazeWindow
from app.runtime import open_session, provider_label


def main() -> None:
    face_session, face_provider = open_session(FACE_FILE)
    gaze_session, gaze_provider = open_session(GAZE_FILE)
    face_input = face_session.get_inputs()[0]
    gaze_input = gaze_session.get_inputs()[0]
    print("face", provider_label(face_provider), face_input.name, face_input.shape)
    print("gaze", provider_label(gaze_provider), gaze_input.name, gaze_input.shape)
    for item in gaze_session.get_outputs():
        print("out", item.name, item.shape)
    estimator = GazeEstimator(gaze_session)
    print("angles", estimator.estimate(np.zeros((224, 224, 3), dtype=np.uint8)))
    engine = GazeEngine(YuNetDetector(face_session), estimator)
    result = engine.detect(
        np.zeros((480, 640, 3), dtype=np.uint8),
        GazeWindow(-30, 5, -22, 22),
        temporal=False,
    )
    print(result)


if __name__ == "__main__":
    main()
