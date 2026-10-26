import unittest

from industrial_fewshot.evaluation.detection_metrics import evaluate_detection, box_iou, check_evaluation_dependencies


def sample(truths=None, predictions=None):
    return {"width": 100, "height": 100, "ground_truth": truths or [], "predictions": predictions or []}


def obj(box=None, score=0.9, label="target"):
    return {"label": label, "bbox_pixels_xyxy": box or [10, 10, 50, 60], "confidence": score}


class DetectionMetricsTest(unittest.TestCase):
    def test_dependency_error_names_install_command(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"numpy": None}):
            with self.assertRaisesRegex(RuntimeError, "python -m pip install numpy scipy"):
                check_evaluation_dependencies()

    def test_perfect_detection_all_metrics_and_zero_loss(self):
        result = evaluate_detection([sample([obj()], [obj()])], ["target"])
        for key in ("mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1", "mean_matched_iou"):
            self.assertAlmostEqual(result[key], 1)
        self.assertEqual(result["loss"]["total"], 0)
        self.assertFalse(result["loss"]["is_model_training_loss"])

    def test_missing_detection_is_false_negative(self):
        result = evaluate_detection([sample([obj()])], ["target"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertEqual(result["mAP50_95"], 0)
        self.assertEqual(result["loss"]["components"]["false_negative"], 1)

    def test_negative_image_and_no_gt_class(self):
        result = evaluate_detection([sample(predictions=[obj()])], ["target"])
        self.assertEqual(result["fp"], 1)
        self.assertIsNone(result["mAP50"])
        self.assertEqual(result["loss"]["total"], 1)

    def test_duplicate_cannot_match_twice(self):
        result = evaluate_detection([sample([obj()], [obj(score=0.9), obj(score=0.8)])], ["target"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["mAP50"], 1)
        self.assertEqual(result["loss"]["total"], 0.5)

    def test_confidence_ranking_changes_ap(self):
        bad = obj([70, 70, 90, 90], score=0.99)
        result = evaluate_detection([sample([obj()], [bad, obj(score=0.8)])], ["target"])
        self.assertAlmostEqual(result["mAP50"], 0.5)

    def test_wrong_class_is_both_miss_and_false_positive(self):
        result = evaluate_detection([sample([obj()], [obj(label="other")])], ["target", "other"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["mAP50"], 0)
        self.assertIsNone(result["per_class"]["other"]["AP50"])

    def test_iou_threshold_sweep(self):
        result = evaluate_detection([sample([obj([0, 0, 100, 100])], [obj([0, 0, 60, 100])])], ["target"])
        self.assertEqual(result["mAP50"], 1)
        self.assertEqual(result["mAP75"], 0)
        self.assertAlmostEqual(result["mAP50_95"], 0.3)
        self.assertAlmostEqual(result["loss"]["components"]["box_l1"], 0.1)
        self.assertAlmostEqual(result["loss"]["components"]["box_iou"], 0.4)

    def test_fixed_score_threshold_does_not_truncate_ap(self):
        result = evaluate_detection([sample([obj()], [obj(score=0.2)])], ["target"], {"score_threshold": 0.5})
        self.assertEqual(result["mAP50"], 1)
        self.assertEqual(result["recall"], 0)

    def test_global_ranking_spans_images(self):
        result = evaluate_detection([sample(predictions=[obj(score=0.99)]), sample([obj()], [obj(score=0.8)])], ["target"])
        self.assertAlmostEqual(result["mAP50"], 0.5)

    def test_half_recall_101_point_interpolation(self):
        result = evaluate_detection([sample([obj(), obj([60, 60, 80, 80])], [obj()])], ["target"])
        self.assertAlmostEqual(result["mAP50"], 51 / 101)

    def test_max_detections_is_respected(self):
        result = evaluate_detection([sample([obj()], [obj([70, 70, 80, 80], 0.99), obj()])], ["target"], {"max_detections_per_image": 1})
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["mAP50"], 0)

    def test_empty_negative_is_zero_loss_and_undefined_ap(self):
        result = evaluate_detection([sample()], ["target"])
        self.assertIsNone(result["mAP50"])
        self.assertEqual(result["loss"]["total"], 0)

    def test_invalid_evaluation_settings(self):
        for settings in ({"iou_threshold": 0}, {"score_threshold": float("nan")}, {"max_detections_per_image": 0}, {"loss_weights": {"box_l1": -1}},
                         {"score_thresholds": "0.5,0.7"}, {"score_thresholds": [True]},
                         {"score_thresholds": [float('nan')]}, {"score_thresholds": [1.1]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                evaluate_detection([sample()], ["target"], settings)

    def test_iou_geometry(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
