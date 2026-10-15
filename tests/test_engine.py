import unittest

import numpy as np

from app.engine import GazeEngine
from app.geometry import GazeWindow
from app.tracker import GazeTracker


WINDOW = GazeWindow(-30, 5, -22, 22)


class ScriptDetector:
    def __init__(self, faces):
        self.faces = list(faces)
        self.shapes = []

    def detect(self, image):
        self.shapes.append(image.shape[:2])
        return self.faces.pop(0)


class FixedGaze:
    def __init__(self, pitch, yaw):
        self.pitch = pitch
        self.yaw = yaw
        self.crops = []

    def estimate(self, crop):
        self.crops.append(crop.copy())
        return self.pitch, self.yaw


class EngineTest(unittest.TestCase):
    def test_scales_largest_face_back_and_crops_original(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[80:240, 80:280] = 255
        detector = ScriptDetector(
            [
                np.array(
                    [
                        [10, 10, 20, 20, 0.9],
                        [40, 30, 50, 80, 0.95],
                    ],
                    dtype=np.float32,
                )
            ]
        )
        gaze = FixedGaze(-10, 4)
        engine = GazeEngine(detector, gaze, GazeTracker())

        result = engine.detect(image, WINDOW, temporal=False)

        self.assertEqual(detector.shapes, [(360, 640)])
        self.assertEqual(result["face_box"], [80.0, 60.0, 100.0, 160.0])
        self.assertEqual(gaze.crops[0].shape[:2], (224, 140))
        self.assertEqual(result["pitch"], -10)
        self.assertEqual(result["yaw"], 4)
        self.assertFalse(result["looking_outside"])
        self.assertEqual(result["message"], "屏幕内")
        self.assertEqual(result["image_width"], 1280)
        self.assertEqual(result["image_height"], 720)

    def test_still_image_does_not_change_camera_tracker(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        face = np.array([[10, 10, 40, 40, 0.9]], dtype=np.float32)
        detector = ScriptDetector([face, face, np.zeros((0, 5), dtype=np.float32)])
        engine = GazeEngine(detector, FixedGaze(0, 40), GazeTracker())

        camera = engine.detect(image, WINDOW, temporal=True)
        self.assertTrue(camera["looking_outside"])
        still = engine.detect(image, WINDOW, temporal=False)
        self.assertTrue(still["looking_outside"])
        held = engine.detect(image, WINDOW, temporal=True)
        self.assertEqual(held["yaw"], 40)
        self.assertEqual(held["message"], "屏幕外")

    def test_no_face_on_still_image(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        detector = ScriptDetector([np.zeros((0, 5), dtype=np.float32)])
        engine = GazeEngine(detector, FixedGaze(0, 0), GazeTracker())
        result = engine.detect(image, WINDOW, temporal=False)
        self.assertIsNone(result["pitch"])
        self.assertFalse(result["looking_outside"])
        self.assertEqual(result["directions"], [])
        self.assertIsNone(result["face_box"])
        self.assertEqual(result["message"], "未检测到人脸")


if __name__ == "__main__":
    unittest.main()
