import tempfile
import unittest
from pathlib import Path

from industrial_fewshot.data.labelme import build_labelme_document, load_ground_truth, read_json, write_json


class LabelMeTest(unittest.TestCase):
    def test_prediction_preserves_geometry_and_optional_confidence(self):
        objects = [{"label": "target", "bbox_pixels_xyxy": [10, 20, 30, 45], "confidence": 0.8}]
        document = build_labelme_document("../../frame.jpg", 100, 60, objects)
        self.assertEqual(document["imagePath"], "../../frame.jpg")
        self.assertIsNone(document["imageData"])
        self.assertEqual(document["shapes"][0]["points"], [[10.0, 20.0], [30.0, 45.0]])
        self.assertEqual(document["shapes"][0]["confidence"], 0.8)

    def test_round_trip_truth_and_atomic_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.json"
            document = build_labelme_document("frame.jpg", 100, 60, [{"label": "target", "bbox_pixels_xyxy": [10, 20, 30, 45]}])
            write_json(path, document)
            self.assertEqual(read_json(path), document)
            truth = load_ground_truth(path, 100, 60, {"target"})
            self.assertEqual(truth, [{"label": "target", "bbox_pixels_xyxy": [10.0, 20.0, 30.0, 45.0]}])
            self.assertFalse(path.with_name("prediction.json.tmp").exists())

    def test_empty_shapes_is_negative_but_other_labels_are_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            for shapes in ([], [{"label": "other", "shape_type": "polygon", "points": []}]):
                write_json(path, {"shapes": shapes})
                self.assertEqual(load_ground_truth(path, 100, 60, {"target"}), [])

    def test_reversed_rectangle_points_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            write_json(path, {"shapes": [{"label": "target", "shape_type": "rectangle", "points": [[80.5, 40.5], [10.25, 20.25]]}]})
            self.assertEqual(load_ground_truth(path, 100, 60, {"target"})[0]["bbox_pixels_xyxy"], [10.25, 20.25, 80.5, 40.5])

    def test_unsupported_or_invalid_target_truth_is_rejected(self):
        cases = [
            {"shape_type": "polygon"},
            {"points": [[-1, 2], [20, 30]]},
            {"points": [[10, 2], [10, 30]]},
            {"points": [[float("nan"), 2], [20, 30]]},
            {"flags": {"ignore": True}},
        ]
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "truth.json"
                shape = {"label": "target", "shape_type": "rectangle", "points": [[10, 2], [20, 30]], **changes}
                # Deliberately malformed GT JSON, independent of our strict writer.
                import json
                path.write_text(json.dumps({"shapes": [shape]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_ground_truth(path, 100, 60, {"target"})
