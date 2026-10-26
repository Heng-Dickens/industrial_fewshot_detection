import asyncio
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import industrial_fewshot.experiments as experiments
import industrial_fewshot.runner as pipeline
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.prompts.references import REFERENCE_MODES
from tests.support import FakeClient, fixture, make_image, make_reference_pair, response


def experiment_fixture(root):
    path, raw = fixture(root)
    raw.update(reference_mode="text_json_crops", max_reference_examples=1,
               max_reference_crops=1, max_reference_negatives=5, ablation={"enabled": True})
    _, ref_json = make_reference_pair(root / "references", "positive", label="target")
    reference = read_json(ref_json)
    reference["flags"] = {"REFERENCE_JSON_SENTINEL": True}
    write_json(ref_json, reference)
    make_image(root / "references/positive_crop_01_target.png", (40, 80))
    make_image(root / "references/negative.jpg")
    truth = read_json(root / "images/frame.json")
    truth["flags"] = {"TEST_TRUTH_SECRET": True}
    write_json(root / "images/frame.json", truth)
    write_json(path, raw)
    return path, raw


class ExperimentsTest(unittest.TestCase):
    def execute(self, path, payloads=None, **kwargs):
        clients = []
        def factory(**_):
            if payloads is None:
                self.fail("cached comparisons must not create clients")
            client = FakeClient(payloads[len(clients)])
            clients.append(client)
            return client
        report = asyncio.run(experiments.run_ablation(pipeline.load_config(path), client_factory=factory, **kwargs))
        return report, clients

    def test_four_group_numbers_payloads_and_artifacts_without_test_truth_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            truth_before = (root / "images/frame.json").read_bytes()
            empty = json.dumps({"annotation_status": "no_target", "reason_code": "no_target", "reason": "no target", "target_count": 0, "objects": []})
            report, clients = self.execute(path, [[response()], [empty], [response()], [response()]])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["executed_inference_count"], 4)
            self.assertEqual([row["mAP50"] for row in report["rows"]], [1, 0, 1, 1])
            self.assertEqual(report["rows"][1]["delta_recall"], -1)
            self.assertEqual(report["rows"][1]["delta_mAP50"], -1)
            self.assertEqual(report["rows"][1]["fn"], 1)
            self.assertEqual(len({t["prompt_sha256"] for g in report["groups"] for t in g["targets"]}), 1)
            self.assertEqual(len({g["targets"][0]["request_signature"] for g in report["groups"]}), 4)
            for index, client in enumerate(clients):
                content = client.requests[0]["messages"][0]["content"]
                text = str(content)
                self.assertNotIn("TEST_TRUTH_SECRET", text)
                self.assertNotIn('"points":[[100,100],[800,450]]', text)
                self.assertEqual("REFERENCE_JSON_SENTINEL" in text, index in (1, 3))
                self.assertEqual(sum(b["type"] == "image_url" for b in content), 2 if index in (2, 3) else 1)
                self.assertEqual(content[-1]["type"], "image_url")
                self.assertEqual(report["groups"][index]["image_paths"], ["frame.jpg"])
            comparison = Path(report["comparison_dir"])
            self.assertEqual(read_json(comparison / "comparison.json")["rows"], report["rows"])
            with (comparison / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[1]["mAP50"]), 0)
            self.assertEqual((root / "images/frame.json").read_bytes(), truth_before)

    def test_repeat_comparison_and_evaluation_changes_reuse_all_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = experiment_fixture(Path(directory))
            first, _ = self.execute(path, [[response()]] * 4)
            raw["evaluation"] = {"score_threshold": 0.95}
            write_json(path, raw)
            repeated, _ = self.execute(path)
            self.assertEqual(first["dataset_sha256"], repeated["dataset_sha256"])
            self.assertEqual(repeated["reused_prediction_count"], 4)
            self.assertEqual(repeated["executed_inference_count"], 0)
            self.assertTrue(all(r["recall"] == 0 for r in repeated["rows"]))

    def test_json_changes_rerun_only_json_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            self.execute(path, [[response()]] * 4)
            reference = read_json(root / "references/positive.json")
            reference["flags"]["changed"] = True
            write_json(root / "references/positive.json", reference)
            report, clients = self.execute(path, [[response()]] * 2)
            self.assertEqual(len(clients), 2)
            self.assertEqual([g["executed_inference_count"] for g in report["groups"]], [0, 1, 0, 1])

    def test_crop_changes_rerun_only_crop_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            self.execute(path, [[response()]] * 4)
            Image.new("RGB", (40, 80), "red").save(root / "references/positive_crop_01_target.png")
            report, _ = self.execute(path, [[response()]] * 2)
            self.assertEqual([g["executed_inference_count"] for g in report["groups"]], [0, 0, 1, 1])

    def test_history_recovers_comparisons_if_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            self.execute(path, [[response()]] * 4)
            cache = (root / "results/.inference_cache").resolve()
            self.assertTrue(cache.is_relative_to(root.resolve()))
            shutil.rmtree(cache)
            report, _ = self.execute(path)
            self.assertEqual(report["reused_prediction_count"], 4)

    def test_missing_crop_stops_all_groups_before_any_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            (root / "references/positive_crop_01_target.png").unlink()
            with self.assertRaisesRegex(ValueError, "裁剪"):
                self.execute(path)
            self.assertFalse((root / "results").exists())

    def test_references_cannot_be_test_truth_even_with_separate_truth_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = experiment_fixture(root)
            raw["dataset"]["ground_truth_dir"] = "references"
            reference = read_json(root / "images/frame.json")
            reference.update(imagePath="../images/frame.jpg")
            write_json(root / "references/frame.json", reference)
            write_json(path, raw)
            with self.assertRaisesRegex(ValueError, "参考标注对应待测图片"):
                self.execute(path)
            self.assertFalse((root / "results").exists())

    def test_review_passes_use_only_reference_json_and_never_test_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = experiment_fixture(Path(directory))
            raw["inference"]["enable_completeness_review"] = True
            write_json(path, raw)
            _, clients = self.execute(path, [[response(), response()]] * 4)
            for index, client in enumerate(clients):
                self.assertEqual(len(client.requests), 2)
                for request in client.requests:
                    payload = str(request["messages"])
                    self.assertNotIn("TEST_TRUTH_SECRET", payload)
                    self.assertEqual("REFERENCE_JSON_SENTINEL" in payload, index in (1, 3))

    def test_dry_run_and_config_driven_cli_never_create_outputs_or_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = pipeline.main(["--config", str(path), "--dry-run"])
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertTrue(report["ablation"])
            self.assertEqual(report["scheduled_inference_count"], 4)
            self.assertEqual([g["targets"][0]["reference_mode"] for g in report["groups"]], list(REFERENCE_MODES))
            self.assertFalse((root / "results").exists())

    def test_failed_requests_are_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = experiment_fixture(Path(directory))
            report, _ = self.execute(path, [[response()], [RuntimeError("offline")], [response()], [response()]])
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["failed_request_count"], 1)
            self.assertEqual(report["rows"][1]["evaluation_status"], "incomplete")

    def test_evaluation_error_preserves_partial_table_and_reuses_completed_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            original = pipeline.evaluate_records
            count = 0
            def failing(*args):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("metric failure")
                return original(*args)
            with patch("industrial_fewshot.runner.evaluate_records", side_effect=failing):
                with self.assertRaisesRegex(RuntimeError, "自动复用"):
                    self.execute(path, [[response()]] * 4)
            partial = read_json(next((root / "results").glob("*/comparison.json")))
            self.assertEqual(partial["status"], "failed")
            self.assertEqual(len(partial["rows"]), 1)
            recovered, clients = self.execute(path, [[response()]] * 2)
            self.assertEqual(len(clients), 2)
            self.assertEqual(recovered["reused_prediction_count"], 2)

    def test_comparison_requires_train_and_rejects_changed_images_between_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = experiment_fixture(root)
            config = pipeline.load_config(path, mode="test")
            with self.assertRaisesRegex(ValueError, "train"):
                asyncio.run(experiments.run_ablation(config))
            original = experiments.run
            async def changed(*args, **kwargs):
                result = await original(*args, **kwargs)
                Image.new("RGB", (1000, 500), "red").save(root / "images/frame.jpg")
                return result
            with patch("industrial_fewshot.experiments.run", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, "图片发生变化"):
                    self.execute(path, [[response()]] * 4)
