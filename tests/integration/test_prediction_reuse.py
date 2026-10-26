import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import industrial_fewshot.runner as pipeline
from tests.support import FakeClient, fixture, make_image, make_reference_pair, response


class PredictionReuseTest(unittest.TestCase):
    def run_config(self, path, client=None, **kwargs):
        factory = (lambda **_: client) if client else (lambda **_: self.fail("cached input must not create a model client"))
        return asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=factory, **kwargs))

    def test_switch_to_default_sibling_reuses_previous_scene_runs(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root, mode="test")
                raw["output_dir"] = "runs/test_scene"
                pipeline.write_json(path, raw)
                with patch("industrial_fewshot.config.PROJECT_ROOT", root):
                    first = self.run_config(path, FakeClient([response()]))
                    old_root = root / "runs/test_scene"
                    if legacy:
                        records = Path(first["run_dir"]) / "records.jsonl"
                        record = json.loads(records.read_text(encoding="utf-8"))
                        record.pop("image_sha256")
                        records.write_text(json.dumps(record) + "\n", encoding="utf-8")
                        cache = (old_root / ".inference_cache").resolve()
                        self.assertTrue(cache.is_relative_to(root.resolve()))
                        shutil.rmtree(cache)
                    before = {p: p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
                    raw.pop("output_dir")
                    pipeline.write_json(path, raw)
                    repeated = self.run_config(path)
                self.assertEqual(Path(repeated["run_dir"]).parent, (root / "images_annotations").resolve())
                self.assertEqual(repeated["executed_inference_count"], 0)
                self.assertEqual(repeated["reused_prediction_count"], 1)
                self.assertEqual(before, {p: p.read_bytes() for p in old_root.rglob("*") if p.is_file()})
                self.assertTrue((root / "images_annotations/.inference_cache").is_dir())

    def test_output_override_reuses_configured_output_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            self.run_config(path, FakeClient([response()]))
            config = pipeline.load_config(path, output_dir=root / "elsewhere")
            report = asyncio.run(pipeline.run(config, client_factory=lambda **_: self.fail("must reuse")))
            self.assertEqual(report["executed_inference_count"], 0)
            self.assertEqual(Path(report["run_dir"]).parent, (root / "elsewhere").resolve())

    def test_unchanged_normal_command_reuses_everything_without_model_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture(Path(directory), mode="test")
            first = FakeClient([response()])
            original = self.run_config(path, first)
            repeated = self.run_config(path)
            self.assertEqual(len(first.requests), 1)
            self.assertEqual(repeated["reused_prediction_count"], 1)
            self.assertEqual(repeated["scheduled_inference_count"], 0)
            self.assertEqual(repeated["executed_inference_count"], 0)
            self.assertNotEqual(original["run_dir"], repeated["run_dir"])
            self.assertTrue((Path(repeated["run_dir"]) / "predictions/frame.json").exists())

    def test_evaluation_failure_then_original_command_only_repeats_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture(Path(directory))
            client = FakeClient([response()])
            with patch("industrial_fewshot.runner.evaluate_records", side_effect=RuntimeError("missing evaluation package")):
                with self.assertRaises(RuntimeError):
                    self.run_config(path, client)
            recovered = self.run_config(path)
            self.assertEqual(recovered["metrics"]["mAP50"], 1)
            self.assertEqual(recovered["executed_inference_count"], 0)
            self.assertEqual(len(client.requests), 1)

    def test_only_failed_or_new_images_are_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            make_image(root / "images/other.jpg", (1000, 500))
            first = FakeClient([response(), RuntimeError("offline")])
            initial = self.run_config(path, first)
            self.assertEqual(initial["failed_request_count"], 1)
            make_image(root / "images/new.jpg", (1000, 500))
            second = FakeClient([response(), response()])
            repeated = self.run_config(path, second)
            self.assertEqual(len(second.requests), 2)
            self.assertEqual(repeated["reused_prediction_count"], 1)
            self.assertEqual(repeated["executed_inference_count"], 2)

    def test_content_change_invalidates_only_changed_image_even_with_preserved_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            make_image(root / "images/other.jpg", (1000, 500))
            self.run_config(path, FakeClient([response(), response()]))
            image = root / "images/frame.jpg"
            before = image.stat()
            Image.new("RGB", (1000, 500), "red").save(image)
            os.utime(image, ns=(before.st_atime_ns, before.st_mtime_ns))
            client = FakeClient([response()])
            repeated = self.run_config(path, client)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(repeated["reused_prediction_count"], 1)

    def test_operational_evaluation_truth_and_mode_changes_do_not_spend_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root, mode="test")
            self.run_config(path, FakeClient([response()]))
            raw["mode"] = "train"
            raw["evaluation"] = {"score_threshold": 0.95}
            raw["inference"].update(concurrency=2, max_attempts=4, timeout_seconds=300)
            pipeline.write_json(path, raw)
            truth = pipeline.read_json(root / "images/frame.json")
            truth["shapes"][0]["points"] = [[100, 100], [400, 450]]
            pipeline.write_json(root / "images/frame.json", truth)
            report = self.run_config(path)
            self.assertEqual(report["executed_inference_count"], 0)
            self.assertEqual(report["metrics"]["mAP50"], 0)
            self.assertEqual(report["metrics"]["settings"]["score_threshold"], 0.95)

    def test_request_changes_invalidate_cache(self):
        for change in ("model", "base_url", "prompt", "review", "temperature", "cache_version", "reference"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root, mode="test")
                if change == "review":
                    raw["inference"]["enable_completeness_review"] = True
                if change == "reference":
                    raw["max_reference_examples"] = 1
                    make_reference_pair(root / "references", "positive", label="target")
                pipeline.write_json(path, raw)
                self.run_config(path, FakeClient([response(), response()]))
                if change == "model":
                    raw["model"]["name"] = "different-model"
                elif change == "base_url":
                    raw["model"]["base_url"] = "http://localhost:9000/v1"
                elif change == "prompt":
                    (root / "prompt.txt").write_text("CHANGED_PROMPT {{label_name}}", encoding="utf-8")
                elif change == "review":
                    (root / "review.txt").write_text("CHANGED_REVIEW {{initial_result}}", encoding="utf-8")
                elif change == "temperature":
                    raw["inference"]["temperature"] = 0.5
                elif change == "cache_version":
                    raw["cache_version"] = "v2"
                elif change == "reference":
                    Image.new("RGB", (640, 360), "red").save(root / "references/positive.jpg")
                pipeline.write_json(path, raw)
                client = FakeClient([response(), response()])
                report = self.run_config(path, client)
                self.assertEqual(report["reused_prediction_count"], 0)
                self.assertEqual(report["executed_inference_count"], 1)
                self.assertEqual(len(client.requests), 2 if change == "review" else 1)

    def test_force_refreshes_and_next_normal_run_uses_refreshed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture(Path(directory), mode="test")
            self.run_config(path, FakeClient([response(confidence=0.9)]))
            client = FakeClient([response(confidence=0.75)])
            forced = self.run_config(path, client, force=True)
            self.assertEqual(forced["executed_inference_count"], 1)
            repeated = self.run_config(path)
            record = pipeline.read_json(Path(repeated["run_dir"]) / "summary.json")["annotations"][0]
            self.assertEqual(record["objects"][0]["confidence"], 0.75)

    def test_legacy_run_without_image_hash_is_automatically_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            first = self.run_config(path, FakeClient([response()]))
            run_dir = Path(first["run_dir"])
            records = run_dir / "records.jsonl"
            record = json.loads(records.read_text(encoding="utf-8"))
            record.pop("image_sha256")
            records.write_text(json.dumps(record) + "\n", encoding="utf-8")
            cache = (root / "results/.inference_cache").resolve()
            self.assertTrue(cache.is_relative_to(root.resolve()))
            shutil.rmtree(cache)
            repeated = self.run_config(path)
            self.assertEqual(repeated["reused_prediction_count"], 1)
            migrated = pipeline.read_json(Path(repeated["run_dir"]) / "summary.json")["annotations"][0]
            self.assertTrue(migrated["legacy_cache_adoption"])

    def test_legacy_image_modified_after_original_run_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            first = self.run_config(path, FakeClient([response()]))
            run_dir = Path(first["run_dir"])
            record = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8"))
            record.pop("image_sha256")
            (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            cache = (root / "results/.inference_cache").resolve()
            self.assertTrue(cache.is_relative_to(root.resolve()))
            shutil.rmtree(cache)
            after = (run_dir / "run.json").stat().st_mtime_ns + 1000000000
            os.utime(root / "images/frame.jpg", ns=(after, after))
            client = FakeClient([response()])
            repeated = self.run_config(path, client)
            self.assertEqual(repeated["executed_inference_count"], 1)

    def test_interrupted_run_reuses_completed_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            make_image(root / "images/other.jpg", (1000, 500))
            original = pipeline.infer_image
            calls = 0
            async def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise asyncio.CancelledError()
                return await original(*args)
            with patch("industrial_fewshot.runner.infer_image", interrupted):
                with self.assertRaises(asyncio.CancelledError):
                    self.run_config(path, FakeClient([response()]))
            client = FakeClient([response()])
            recovered = self.run_config(path, client)
            self.assertEqual(recovered["reused_prediction_count"], 1)
            self.assertEqual(len(client.requests), 1)

    def test_dry_run_reports_reuse_without_writing_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            self.run_config(path, FakeClient([response()]))
            before = set((root / "results").iterdir())
            report = self.run_config(path, dry_run=True)
            self.assertEqual(report["scheduled_inference_count"], 0)
            self.assertEqual(set((root / "results").iterdir()), before)

    def test_concurrent_same_inputs_issue_one_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture(Path(directory), mode="test")
            clients = []
            def factory(**kwargs):
                client = FakeClient([response()])
                clients.append(client)
                return client
            async def concurrent():
                return await asyncio.wait_for(asyncio.gather(
                    pipeline.run(pipeline.load_config(path), client_factory=factory),
                    pipeline.run(pipeline.load_config(path), client_factory=factory),
                ), timeout=10)
            reports = asyncio.run(concurrent())
            self.assertEqual(sum(len(client.requests) for client in clients), 1)
            self.assertEqual(sum(report["reused_prediction_count"] for report in reports), 1)
