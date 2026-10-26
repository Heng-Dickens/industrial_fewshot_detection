import unittest

from industrial_fewshot.evaluation.evaluator import evaluate_records


def obj(box=None, score=0.9, label="target"):
    return {"label": label, "confidence": score, "bbox_pixels_xyxy": box or [10, 10, 50, 50]}


def image(path, truths):
    return {"relative_path": path, "width": 100, "height": 100, "ground_truth": truths}


def result(path, predictions=(), status="success", target="sample"):
    return {"relative_path": path, "target_id": target, "status": status, "objects": list(predictions)}


class ImageAccuracyTest(unittest.TestCase):
    def test_failed_negative_reduces_overall_accuracy_and_negative_error_rate(self):
        metrics = evaluate_records(
            [image("positive", [obj()]), image("negative", []), image("failed", [])],
            [result("positive", [obj()]), result("negative"), result("failed", status="failed")], ["target"], {})
        self.assertEqual(metrics["image_accuracy"], 2 / 3)
        self.assertEqual(metrics["correct_image_count"], 2)
        self.assertEqual(metrics["incorrect_image_count"], 1)
        self.assertEqual(metrics["failed_image_count"], 1)
        self.assertEqual(metrics["negative_error_rate"], 0.5)
        self.assertEqual(metrics["negative_false_positive_rate"], 0)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["incorrect_images"], [{"relative_path": "failed", "reason": "inference_failed"}])

    def test_failed_positive_is_an_error_and_retains_false_negative_boxes(self):
        metrics = evaluate_records([image("failed", [obj()])], [result("failed", status="failed")], ["target"], {})
        self.assertEqual(metrics["image_accuracy"], 0)
        self.assertEqual(metrics["incorrect_image_count"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["negative_error_rate"], None)

    def test_all_failed_negatives_have_zero_accuracy_and_full_error_rate(self):
        metrics = evaluate_records([image("failed", [])], [result("failed", status="failed")], ["target"], {})
        self.assertEqual(metrics["image_accuracy"], 0)
        self.assertEqual(metrics["negative_error_rate"], 1)
        self.assertEqual(metrics["negative_error_image_count"], 1)
        self.assertEqual(metrics["negative_true_negative_image_count"], 0)
        self.assertIsNone(metrics["negative_false_positive_rate"])

    def test_successful_empty_negative_is_correct_but_spurious_box_is_wrong(self):
        metrics = evaluate_records([image("empty", []), image("spurious", [])],
                                   [result("empty"), result("spurious", [obj()])], ["target"], {})
        self.assertEqual(metrics["image_accuracy"], 0.5)
        self.assertEqual(metrics["negative_error_rate"], 0.5)

    def test_missing_extra_or_wrong_class_box_makes_entire_image_incorrect(self):
        cases = [
            ([obj(), obj([55, 55, 95, 95])], [obj()]),
            ([obj()], [obj(), obj(score=0.8)]),
            ([obj()], [obj(label="other")]),
        ]
        for truths, predictions in cases:
            with self.subTest(truths=truths, predictions=predictions):
                metrics = evaluate_records([image("frame", truths)], [result("frame", predictions)], ["target", "other"], {})
                self.assertEqual(metrics["image_accuracy"], 0)
                self.assertEqual(metrics["incorrect_images"][0]["reason"], "detection_error")

    def test_image_accuracy_follows_iou_and_score_matching_settings(self):
        for threshold, expected in ((0.5, 1), (0.75, 0)):
            metrics = evaluate_records([image("frame", [obj()])],
                [result("frame", [obj([20, 10, 60, 50]), obj(score=0.2)])],
                ["target"], {"iou_threshold": threshold, "score_threshold": 0.5})
            self.assertEqual(metrics["image_accuracy"], expected)

    def test_multiple_failed_targets_count_as_one_incorrect_image(self):
        metrics = evaluate_records([image("frame", [])],
            [result("frame", status="failed", target="a"), result("frame", status="failed", target="b")], ["target"], {})
        self.assertEqual(metrics["failed_request_count"], 2)
        self.assertEqual(metrics["failed_image_count"], 1)
        self.assertEqual(metrics["incorrect_image_count"], 1)
        self.assertEqual(metrics["image_accuracy"], 0)
