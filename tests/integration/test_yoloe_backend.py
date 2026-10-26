import asyncio
import copy
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import industrial_fewshot.runner as pipeline
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.data.labelme import read_json, write_json, validate_pixel_prediction
from industrial_fewshot.prompts.references import prepare_reference_examples
from tests.integration.test_reference_suite import suite_fixture
from tests.support import FakeClient, response
from industrial_fewshot.detectors.yoloe_visual import YOLOERunner, yoloe_settings
from industrial_fewshot.evaluation.compare import compare


def local_fixture(root):
    path, raw = suite_fixture(root, positive_counts=[5])
    raw.update(backend="yoloe", experiment_suite={"enabled": False},
               yoloe={"source_dir": "checkout", "weights": "weights.pt"})
    source = root / "checkout/ultralytics/models/yolo/yoloe/predict_vp.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Fake source for interface tests", encoding="utf-8")
    write_json(root / "checkout/UPSTREAM.json", {"revision": "test_revision"})
    (root / "weights.pt").write_bytes(b"FAKE_FOR_MOCK_TESTS_ONLY")
    write_json(path, raw)
    return path, raw


class FakeRunner:
    setup_seconds = 0.01
    reference_encoding_seconds = 0.02

    def __init__(self, config):
        self.config, self.calls, self.closed = config, [], False
        if "TEST_TRUTH_SECRET" in json.dumps(config):
            raise AssertionError("Test GT leaked to local detector config")

    def predict(self, data):
        self.calls.append(data)
        # Test images are never accompanied by their evaluation truth.
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (1000, 500):
                raise AssertionError("Wrong test image")
        return {"annotation_status": "annotated", "reason_code": "target_detected", "reason": "fake",
                "target_count": 1, "objects": [{"label": "target", "confidence": 0.9,
                 "bbox_pixels_xyxy": [100.125, 100.25, 800.5, 450.125]}]}

    async def close(self):
        self.closed = True


class YOLOEBackendTest(unittest.TestCase):
    def execute(self, path, dry_run=False, forbid=False, runner_type=FakeRunner):
        runners = []
        def factory(config):
            if forbid:
                self.fail("Fully cached/dry run must not initialize a local model")
            runner = runner_type(config)
            runners.append(runner)
            return runner
        with patch("industrial_fewshot.runner.inference.request_annotation", side_effect=AssertionError("No VLM API")):
            report = asyncio.run(pipeline.run(pipeline.load_config(path), dry_run=dry_run, local_runner_factory=factory))
        return report, runners

    def test_dry_run_defaults_five_references_without_api_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = local_fixture(root)
            report, runners = self.execute(path, dry_run=True, forbid=True)
            self.assertFalse(runners)
            self.assertFalse((root / "results").exists())
            descriptor = report["targets"][0]
            self.assertEqual(descriptor["reference_count"]["reference_originals"], 5)
            self.assertEqual(descriptor["reference_count"]["reference_negatives"], 0)
            self.assertEqual(descriptor["backend"], "yoloe")
            self.assertIsNone(descriptor["base_url"])
            self.assertEqual(descriptor["local_model"]["upstream_revision"], "test_revision")

    def test_float_precision_saved_cache_and_evaluate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = local_fixture(Path(directory))
            first, runners = self.execute(path)
            self.assertEqual(len(runners), 1)
            self.assertTrue(runners[0].closed)
            self.assertEqual(first["metrics"]["image_accuracy"], 1)
            self.assertEqual(first["timing"]["fresh_timed_image_count"], 1)
            summary = read_json(Path(first["run_dir"]) / "summary.json")
            self.assertEqual(summary["annotations"][0]["objects"][0]["bbox_pixels_xyxy"], [100.125, 100.25, 800.5, 450.125])
            repeated, _ = self.execute(path, forbid=True)
            self.assertEqual(repeated["reused_prediction_count"], 1)
            self.assertEqual(repeated["timing"]["fresh_timed_image_count"], 0)
            recovered = evaluate_saved_run(Path(first["run_dir"]))
            self.assertEqual(recovered["metrics"]["mAP50"], first["metrics"]["mAP50"])

    def test_evaluation_changes_reuse_but_model_and_used_box_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = local_fixture(root)
            first, _ = self.execute(path)
            raw["evaluation"] = {"score_threshold": 0.95}
            write_json(path, raw)
            evaluation, _ = self.execute(path, forbid=True)
            self.assertEqual(evaluation["metrics"]["recall"], 0)
            self.assertEqual(evaluation['reused_prediction_count'], 1)
            self.assertEqual(first['targets'][0]['request_signature'], evaluation['targets'][0]['request_signature'])
            raw['yoloe']['confidence'] = 0.2
            write_json(path, raw)
            detection, runners = self.execute(path)
            self.assertEqual(detection['executed_inference_count'], 1)
            self.assertEqual(runners[0].config['yoloe']['confidence'], 0.2)
            self.assertNotEqual(evaluation['targets'][0]['request_signature'], detection['targets'][0]['request_signature'])
            (root / "weights.pt").write_bytes(b"CHANGED_FAKE_WEIGHTS")
            weights, _ = self.execute(path)
            self.assertNotEqual(first["targets"][0]["request_signature"], weights["targets"][0]["request_signature"])
            config = pipeline.load_config(path)
            prepare_reference_examples(config["targets"][0]["config"])
            ref_path = Path(config["targets"][0]["config"]["reference_examples"][0]["json_path"])
            document = read_json(ref_path)
            document["flags"]["unused_metadata"] = True
            write_json(ref_path, document)
            self.execute(path, forbid=True)
            document["shapes"][0]["points"][0][0] += 0.125
            write_json(ref_path, document)
            boxes, _ = self.execute(path)
            self.assertEqual(boxes["executed_inference_count"], 1)
            self.assertNotEqual(weights["targets"][0]["request_signature"], boxes["targets"][0]["request_signature"])

    def test_failures_remain_errors_and_are_retried_next_run(self):
        class FailedRunner(FakeRunner):
            def predict(self, data):
                raise RuntimeError("offline local detector")
        with tempfile.TemporaryDirectory() as directory:
            path, _ = local_fixture(Path(directory))
            first, _ = self.execute(path, runner_type=FailedRunner)
            self.assertEqual(first["metrics"]["failed_image_count"], 1)
            self.assertEqual(first["metrics"]["image_accuracy"], 0)
            recovered, _ = self.execute(path)
            self.assertEqual(recovered["executed_inference_count"], 1)

    def test_disjoint_cohort_matches_vlm_and_overlap_can_be_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = local_fixture(root)
            Image.open(root / "prepared/positive/images/p00.png").save(root / "images/copy.png")
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            report, _ = self.execute(path, dry_run=True)
            self.assertEqual(report["excluded_reference_overlap"], ["copy.png"])
            self.assertEqual(report["image_paths"], ["frame.jpg"])
            raw["yoloe"]["exclude_reference_overlap"] = False
            write_json(path, raw)
            with self.assertRaisesRegex(ValueError, "内容重复"):
                self.execute(path, dry_run=True)

    def test_saved_detector_comparison_recomputes_one_cohort_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = local_fixture(root)
            local, _ = self.execute(path)
            raw.update(backend="generic", reference_mode="text_only", output_dir="vlm_results")
            write_json(path, raw)
            vlm = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: FakeClient([response()])))
            report = compare([Path(vlm["run_dir"])], Path(local["run_dir"]), root / "comparison")
            self.assertEqual(report["image_count"], 1)
            self.assertEqual([r["method"] for r in report["rows"]], ["YOLOE", "VLM"])
            self.assertTrue(all(r["mAP50"] == 1 for r in report["rows"]))
            self.assertTrue((root / "comparison/detector_comparison.csv").is_file())
            Image.new("RGB", (1000, 500), "red").save(root / "images/frame.jpg")
            with self.assertRaisesRegex(ValueError, "图片已改变"):
                compare([Path(vlm["run_dir"])], Path(local["run_dir"]), root / "invalid_comparison")
            self.assertFalse((root / "invalid_comparison").exists())

    def test_pixel_validator_and_local_settings_reject_bad_data(self):
        for settings in ({"positive_count": True}, {"positive_count": 0}, {"imgsz": 641},
                         {"confidence": float("nan")}, {"aggregation": "first"}, {"negative_count": 5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                yoloe_settings(settings, Path.cwd())
        valid = {"annotation_status": "annotated", "reason_code": "target_detected", "reason": "fake",
                 "target_count": 1, "objects": [{"label": "target", "confidence": 0.9,
                  "bbox_pixels_xyxy": [0.125, 1.25, 1.125, 2.25]}]}
        checked = validate_pixel_prediction(valid, {"target"}, 10000, 10000)
        self.assertEqual(checked["objects"][0]["bbox_pixels_xyxy"], valid["objects"][0]["bbox_pixels_xyxy"])
        for box in ([0, 0, 0, 1], [0, 0, float("nan"), 2], [0, 0, 10001, 2]):
            broken = copy.deepcopy(valid)
            broken["objects"][0]["bbox_pixels_xyxy"] = box
            with self.assertRaises(ValueError):
                validate_pixel_prediction(broken, {"target"}, 10000, 10000)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for prototype math test")
    def test_official_api_uses_all_five_references_then_normalized_mean_for_target(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = local_fixture(root)
            config = pipeline.load_config(path)["targets"][0]["config"]
            prepare_reference_examples(config)
            package = types.ModuleType("ultralytics")
            package.__file__ = str(root / "checkout/ultralytics/__init__.py")
            predictor_module = types.ModuleType("ultralytics.models.yolo.yoloe.predict_vp")
            predictor_module.YOLOEVPSegPredictor = type("Predictor", (), {})
            vectors = [torch.tensor([[[float(i + 1), 1.0, 2.0]]]) for i in range(5)]
            class Model:
                def __init__(self, weights):
                    self.model = torch.nn.Linear(1, 1)
                    self.predictor, self.reference_calls, self.target_calls = None, [], []
                def predict(self, image, **kwargs):
                    if "prompts" in kwargs:
                        self.predictor = types.SimpleNamespace(vpe=vectors[len(self.reference_calls)])
                        self.reference_calls.append(kwargs)
                        return []
                    self.target_calls.append(kwargs)
                    return [types.SimpleNamespace(boxes=None)]
                def set_classes(self, names, embeddings):
                    self.names, self.embeddings = names, embeddings
            package.YOLOE = Model
            with patch.dict("sys.modules", {"ultralytics": package,
                                           "ultralytics.models.yolo.yoloe.predict_vp": predictor_module}):
                runner = YOLOERunner(config)
                expected = torch.nn.functional.normalize(torch.stack([
                    torch.nn.functional.normalize(v, dim=-1) for v in vectors]).mean(dim=0), dim=-1)
                self.assertTrue(torch.allclose(runner.model.embeddings, expected))
                self.assertEqual(len(runner.model.reference_calls), 5)
                self.assertTrue(all(set(call["prompts"]["cls"]) == {0} for call in runner.model.reference_calls))
                result = runner.predict((root / "images/frame.jpg").read_bytes())
                self.assertEqual(result["objects"], [])
                self.assertNotIn("prompts", runner.model.target_calls[0])
