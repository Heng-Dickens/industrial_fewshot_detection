import importlib.util
from pathlib import Path
import unittest

import numpy as np

from industrial_fewshot.detectors.yolo_onnx import decode_heads, letterbox, yolo_onnx_settings


class YOLOONNXUnitTest(unittest.TestCase):
    def settings(self, **changes):
        return yolo_onnx_settings(dict(weights='model.onnx', input_size=[64, 64], reg_max=4, **changes), Path.cwd())

    def heads(self):
        return [np.full((1, 17, 64 // stride, 64 // stride), -80, dtype=np.float32)
                for stride in (8, 16, 32)]

    def candidate(self, head, x, y, probability):
        for side in range(4):
            head[0, side * 4 + 2, y, x] = 80
        head[0, 16, y, x] = np.log(probability / (1 - probability))

    def test_raw_dfl_sigmoid_nms_and_inverse_letterbox_coordinates(self):
        heads = self.heads()
        self.candidate(heads[0], 3, 3, 0.9)  # anchor28, distances16 -> [12,12,44,44]
        self.candidate(heads[1], 1, 1, 0.8)  # overlapping lower-score candidate
        # Different distances for second candidate retain >0.5 IoU with the first.
        heads[1][0, :16, 1, 1] = -80
        for side in range(4):
            heads[1][0, side * 4 + 1, 1, 1] = 80
        result = decode_heads(heads[::-1], self.settings(nms_iou=0.5), (2, 4, 8, 40, 32), 'target')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox_pixels_xyxy'], [4, 2, 20, 18])
        self.assertAlmostEqual(result[0]['confidence'], 0.9, places=6)

    def test_empty_outputs_and_letterbox_only_detection_are_discarded(self):
        heads = self.heads()
        self.assertEqual(decode_heads(heads, self.settings(), (1, 0, 0, 64, 64), 'target'), [])
        self.candidate(heads[0], 1, 1, 0.8)
        self.assertEqual(decode_heads(heads, self.settings(), (1, 0, 40, 64, 8), 'target'), [])

    def test_bad_heads_raise_instead_of_silent_misdecoding(self):
        for kind in ('nan', 'wrong_channels', 'missing_scale', 'duplicate_scale'):
            with self.subTest(kind=kind):
                heads = self.heads()
                if kind == 'nan':
                    heads[0][0, 0, 0, 0] = np.nan
                elif kind == 'wrong_channels':
                    heads[0] = np.zeros((1, 18, 8, 8), dtype=np.float32)
                elif kind == 'missing_scale':
                    heads.pop()
                else:
                    heads[1] = heads[0]
                with self.assertRaises(ValueError):
                    decode_heads(heads, self.settings(), (1, 0, 0, 64, 64), 'target')

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV dependency')
    def test_letterbox_preserves_rgb_layout_and_odd_padding(self):
        image = np.zeros((101, 200, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        tensor, transform = letterbox(image, [960, 1280])
        self.assertEqual(tensor.shape, (1, 3, 960, 1280))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(transform, (6.4, 0, 157, 200, 101))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 114 / 255, places=6)
        self.assertEqual(tensor[0, :, 200, 100].tolist(), [1, 0, 0])

    def test_configuration_rejects_ambiguous_or_invalid_model_settings(self):
        cases = [dict(weights='model.pt'), dict(num_classes=2), dict(class_names=[]),
                 dict(target_class_id=True), dict(target_class_id=1), dict(input_size=[640]),
                 dict(input_size=[961, 1280]), dict(confidence=float('nan')), dict(reg_max=1),
                 dict(providers=[]), dict(exclude_reference_overlap=1)]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                yolo_onnx_settings({'weights': 'model.onnx', **change}, Path.cwd())
