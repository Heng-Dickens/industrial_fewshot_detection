import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import industrial_fewshot.runner as pipeline
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.data.labelme import read_json, write_json
from tests.integration.test_experiments import experiment_fixture
from tests.support import FakeClient, fixture, make_image, response


def empty_response():
    return json.dumps({"annotation_status": "no_target", "reason_code": "no_target",
                       "reason": "no target", "target_count": 0, "objects": []})


class NegativeEvaluationTest(unittest.TestCase):
    def execute(self, path, payloads=None):
        client = FakeClient(payloads) if payloads is not None else None
        def factory(**_):
            if client is None:
                self.fail("evaluation policy change must reuse inference")
            return client
        return asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=factory))

    def test_missing_annotations_count_false_positives_and_true_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            make_image(root / "images/a_negative.jpg", (1000, 500))
            make_image(root / "images/b_negative.jpg", (1000, 500))
            report = self.execute(path, [response(confidence=0.95), empty_response(), response()])
            metrics = report["metrics"]
            self.assertEqual(metrics["evaluated_image_count"], 3)
            self.assertEqual(metrics["ground_truth_json_image_count"], 1)
            self.assertEqual(metrics["assumed_negative_image_count"], 2)
            self.assertEqual(metrics["unlabeled_image_count"], 0)
            self.assertEqual(metrics["tp"], 1)
            self.assertEqual(metrics["fp"], 1)
            self.assertEqual(metrics["precision"], 0.5)
            self.assertEqual(metrics["recall"], 1)
            self.assertEqual(metrics["mAP50"], 0.5)
            self.assertEqual(metrics["negative_image_count"], 2)
            self.assertEqual(metrics["negative_false_positive_image_count"], 1)
            self.assertEqual(metrics["negative_true_negative_image_count"], 1)
            self.assertEqual(metrics["negative_false_positive_rate"], 0.5)
            self.assertEqual(metrics["negative_false_positive_box_count"], 1)
            self.assertEqual(metrics["negative_false_positive_images"][0]["relative_path"], "a_negative.jpg")
            self.assertFalse((root / "images/a_negative.json").exists())

    def test_policy_change_reuses_predictions_and_can_override_saved_run_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            make_image(root / "images/negative.jpg", (1000, 500))
            original = self.execute(path, [response(), response()])
            self.assertEqual(original["metrics"]["fp"], 0)
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            records = Path(original["run_dir"]) / "records.jsonl"
            before = records.read_bytes()
            recovered = evaluate_saved_run(Path(original["run_dir"]), path)
            self.assertEqual(recovered["missing_ground_truth"], "negative")
            self.assertEqual(recovered["metrics"]["fp"], 1)
            self.assertEqual(records.read_bytes(), before)
            repeated = self.execute(path)
            self.assertEqual(repeated["reused_prediction_count"], 2)
            self.assertEqual(repeated["executed_inference_count"], 0)
            self.assertEqual(repeated["metrics"]["fp"], 1)

    def test_entirely_negative_dataset_is_evaluable_without_any_gt_json(self):
        for payload, fp, tn in ((response(), 1, 0), (empty_response(), 0, 1)):
            with self.subTest(fp=fp), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root)
                raw["dataset"]["missing_ground_truth"] = "negative"
                write_json(path, raw)
                (root / "images/frame.json").unlink()
                report = self.execute(path, [payload])
                metrics = report["metrics"]
                self.assertEqual(metrics["negative_false_positive_image_count"], fp)
                self.assertEqual(metrics["negative_true_negative_image_count"], tn)
                self.assertEqual(metrics["negative_false_positive_rate"], float(fp))
                self.assertIsNone(metrics["mAP50"])
                self.assertEqual(metrics["ground_truth_json_image_count"], 0)
                saved = evaluate_saved_run(Path(report["run_dir"]))
                self.assertEqual(saved["metrics"]["negative_false_positive_rate"], float(fp))

    def test_failed_negative_is_not_a_true_negative_or_zero_false_positive_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            make_image(root / "images/negative.jpg", (1000, 500))
            report = self.execute(path, [response(), RuntimeError("offline")])
            metrics = report["metrics"]
            self.assertEqual(metrics["evaluation_status"], "incomplete")
            self.assertEqual(metrics["negative_failed_image_count"], 1)
            self.assertEqual(metrics["negative_true_negative_image_count"], 0)
            self.assertIsNone(metrics["negative_false_positive_rate"])
            self.assertEqual(metrics["image_accuracy"], 0.5)
            self.assertEqual(metrics["failed_image_count"], 1)
            self.assertEqual(metrics["negative_error_rate"], 1)

    def test_negative_rate_uses_evaluation_score_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["dataset"]["missing_ground_truth"] = "negative"
            raw["evaluation"] = {"score_threshold": 0.95}
            write_json(path, raw)
            make_image(root / "images/negative.jpg", (1000, 500))
            report = self.execute(path, [response(confidence=0.99), response(confidence=0.9)])
            self.assertEqual(report["metrics"]["fp"], 0)
            self.assertEqual(report["metrics"]["negative_false_positive_rate"], 0)
            self.assertEqual(report["metrics"]["negative_true_negative_image_count"], 1)

    def test_explicit_empty_json_and_missing_negative_are_both_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            write_json(root / "images/frame.json", {"shapes": []})
            make_image(root / "images/negative.jpg", (1000, 500))
            report = self.execute(path, [empty_response(), response()])
            self.assertEqual(report["metrics"]["negative_image_count"], 2)
            self.assertEqual(report["metrics"]["assumed_negative_image_count"], 1)
            self.assertEqual(report["metrics"]["negative_false_positive_rate"], 0.5)

    def test_invalid_json_still_fails_instead_of_becoming_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            (root / "images/frame.json").write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.execute(path)
            self.assertFalse((root / "results").exists())

    def test_test_mode_does_not_read_truth_even_under_negative_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root, mode="test")
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            (root / "images/frame.json").write_text("{broken", encoding="utf-8")
            report = self.execute(path, [response()])
            self.assertNotIn("metrics", report)
            self.assertEqual(report["success_request_count"], 1)

    def test_four_group_table_includes_negative_errors_and_policy_change_reuses_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = experiment_fixture(root)
            make_image(root / "images/negative.jpg", (1000, 500))
            def factory(**_):
                return FakeClient([response(), response()])
            first = asyncio.run(run_ablation(pipeline.load_config(path), client_factory=factory))
            self.assertTrue(all(row["fp"] == 0 for row in first["rows"]))
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            report = asyncio.run(run_ablation(pipeline.load_config(path),
                client_factory=lambda **_: self.fail("all four groups must reuse")))
            self.assertEqual(report["executed_inference_count"], 0)
            self.assertEqual(report["reused_prediction_count"], 8)
            self.assertEqual(report["assumed_negative_image_count"], 1)
            for row in report["rows"]:
                self.assertEqual(row["fp"], 1)
                self.assertEqual(row["negative_false_positive_rate"], 1)
                self.assertEqual(row["negative_false_positive_image_count"], 1)
                self.assertEqual(row["assumed_negative_image_count"], 1)
                self.assertEqual(row["delta_negative_false_positive_rate"], 0)
                self.assertEqual(row["image_accuracy"], 0.5)
                self.assertEqual(row["negative_error_rate"], 1)
                self.assertEqual(row["delta_image_accuracy"], 0)
            csv_text = (Path(report["comparison_dir"]) / "comparison.csv").read_text(encoding="utf-8-sig")
            self.assertIn("negative_false_positive_rate", csv_text)
