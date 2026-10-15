import unittest

import numpy as np

from app.face_detector import decode_yunet, nms_xywh


class FaceDetectorTest(unittest.TestCase):
    def test_nms_suppresses_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [40, 40, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(nms_xywh(boxes, scores, 0.3, 10), [0, 2])

    def test_decode_yunet_one_cell(self):
        outputs = {}
        for stride, rows in ((8, 4), (16, 2), (32, 1)):
            outputs[f"cls_{stride}"] = np.zeros((1, 1, rows, rows), dtype=np.float32)
            outputs[f"obj_{stride}"] = np.zeros((1, 1, rows, rows), dtype=np.float32)
            outputs[f"bbox_{stride}"] = np.zeros((1, 4, rows, rows), dtype=np.float32)
        outputs["cls_32"][0, 0, 0, 0] = 1
        outputs["obj_32"][0, 0, 0, 0] = 1
        faces = decode_yunet(outputs, pad_w=32, pad_h=32, score_threshold=0.6)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(float(faces[0, 0]), -16.0, places=4)
        self.assertAlmostEqual(float(faces[0, 2]), 32.0, places=4)
        self.assertAlmostEqual(float(faces[0, 4]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
