import unittest

import numpy as np

from app.geometry import (
    GazeWindow,
    gaze_directions,
    limit_long_edge,
    next_outside,
    parse_range,
    raw_outside,
)
from app.gaze_decode import logits_to_degrees, to_screen_angles
from app.tracker import GazeTracker


WINDOW = GazeWindow(-30, 5, -22, 22)


class WindowTest(unittest.TestCase):
    def test_parse_range(self):
        self.assertEqual(parse_range("-30,5"), (-30.0, 5.0))
        with self.assertRaises(ValueError):
            parse_range("5,-30")

    def test_raw_outside_uses_window_edges(self):
        self.assertFalse(raw_outside(-30, -22, WINDOW))
        self.assertFalse(raw_outside(5, 22, WINDOW))
        self.assertTrue(raw_outside(-30.1, 0, WINDOW))
        self.assertTrue(raw_outside(0, 22.1, WINDOW))

    def test_directions_follow_window_edges(self):
        self.assertEqual(gaze_directions(6, 0, WINDOW), ["上"])
        self.assertEqual(gaze_directions(-31, 0, WINDOW), ["下"])
        self.assertEqual(gaze_directions(0, -23, WINDOW), ["左"])
        self.assertEqual(gaze_directions(0, 23, WINDOW), ["右"])
        self.assertEqual(gaze_directions(6, 23, WINDOW), ["上", "右"])
        self.assertEqual(gaze_directions(0, 0, WINDOW), [])

    def test_enter_and_leave_use_three_degree_hysteresis(self):
        self.assertFalse(next_outside(False, -33, 0, WINDOW))
        self.assertTrue(next_outside(False, -33.1, 0, WINDOW))
        self.assertTrue(next_outside(False, 8.1, 0, WINDOW))
        self.assertTrue(next_outside(False, 0, 25.1, WINDOW))
        self.assertTrue(next_outside(True, -28, 0, WINDOW))
        self.assertTrue(next_outside(True, -32, 0, WINDOW))
        self.assertFalse(next_outside(True, -27, 0, WINDOW))
        self.assertFalse(next_outside(True, 2, 19, WINDOW))

    def test_limit_long_edge(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        resized, scale = limit_long_edge(image, 640)
        self.assertEqual(resized.shape[:2], (360, 640))
        self.assertAlmostEqual(scale, 0.5)
        small = np.zeros((80, 100, 3), dtype=np.uint8)
        same, same_scale = limit_long_edge(small, 640)
        self.assertEqual(same.shape[:2], (80, 100))
        self.assertEqual(same_scale, 1.0)


class DecodeTest(unittest.TestCase):
    def test_bin_expectation_is_degrees(self):
        logits = np.full(90, -100.0, dtype=np.float32)
        logits[45] = 10
        self.assertAlmostEqual(logits_to_degrees(logits), 0.0, places=4)
        logits = np.full(90, -100.0, dtype=np.float32)
        logits[0] = 10
        self.assertAlmostEqual(logits_to_degrees(logits), -180.0, places=4)

    def test_positive_yaw_points_to_image_right(self):
        pitch, yaw = to_screen_angles(12.0, 8.0)
        self.assertAlmostEqual(pitch, 12.0)
        self.assertAlmostEqual(yaw, -8.0)


class TrackerTest(unittest.TestCase):
    def test_still_frame_ignores_hysteresis_and_history(self):
        tracker = GazeTracker()
        outside = tracker.classify_still(-33, 0, WINDOW)
        self.assertTrue(outside["looking_outside"])
        self.assertEqual(outside["directions"], ["下"])
        inside = tracker.classify_still(-30, 0, WINDOW)
        self.assertFalse(inside["looking_outside"])
        self.assertEqual(inside["message"], "屏幕内")
        followed = tracker.update(0, 0, WINDOW)
        self.assertEqual(followed["pitch"], 0)
        self.assertFalse(followed["looking_outside"])

    def test_camera_uses_median_and_keeps_hysteresis_when_window_changes(self):
        tracker = GazeTracker()
        for yaw in (10, 12, 14, 16, 30):
            result = tracker.update(0, yaw, WINDOW)
        self.assertEqual(result["yaw"], 14)
        self.assertFalse(result["looking_outside"])
        tight = GazeWindow(-30, 5, -22, 10)
        jumped = tracker.update(0, 14, tight)
        self.assertTrue(jumped["looking_outside"])
        self.assertEqual(jumped["directions"], ["右"])

    def test_camera_holds_last_face_until_ten_misses(self):
        tracker = GazeTracker()
        seen = tracker.update(-10, 0, WINDOW)
        for _ in range(9):
            held = tracker.update(None, None, WINDOW)
            self.assertEqual(held["pitch"], seen["pitch"])
            self.assertEqual(held["message"], "屏幕内")
        cleared = tracker.update(None, None, WINDOW)
        self.assertIsNone(cleared["pitch"])
        self.assertIsNone(cleared["yaw"])
        self.assertFalse(cleared["looking_outside"])
        self.assertEqual(cleared["directions"], [])
        self.assertEqual(cleared["message"], "未检测到人脸")

    def test_reset_drops_camera_history(self):
        tracker = GazeTracker()
        tracker.update(40, 40, WINDOW)
        tracker.reset()
        cleared = tracker.update(None, None, WINDOW)
        self.assertEqual(cleared["message"], "未检测到人脸")


if __name__ == "__main__":
    unittest.main()
