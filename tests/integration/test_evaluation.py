import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import industrial_fewshot.runner as pipeline
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.data.labelme import build_labelme_document, read_json, write_json
from tests.support import FakeClient, fixture, response


def saved_run(root: Path):
    config, _ = fixture(root)
    run_dir = root / "results/old_run"
    write_json(run_dir / "run.json", {
        "scene_name": "saved_scene", "mode": "train", "image_dir": str(root / "images"),
        "image_count": 1, "targets": [{"id": "sample", "label": "target"}],
    })
    record = {**json.loads(response()), "target_id": "sample", "relative_path": "frame.jpg", "status": "success"}
    record["objects"][0]["bbox_pixels_xyxy"] = [100, 100, 800, 450]
    (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    write_json(run_dir / "sample/predictions/frame.json", build_labelme_document("frame.jpg", 1000, 500, record["objects"]))
    return config, run_dir


class EvaluationRecoveryTest(unittest.TestCase):
    def test_legacy_run_without_summary_can_be_evaluated_without_touching_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, run_dir = saved_run(root)
            paths = [run_dir / "run.json", run_dir / "records.jsonl", run_dir / "sample/predictions/frame.json", root / "images/frame.json"]
            before = {path: path.read_bytes() for path in paths}
            report = evaluate_saved_run(run_dir)
            self.assertEqual(report["metrics"]["mAP50_95"], 1)
            self.assertEqual(report["metrics"]["loss"]["total"], 0)
            self.assertTrue(report["evaluation_only"])
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_cli_evaluation_does_not_load_model_config_or_execute_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            config, run_dir = saved_run(Path(directory))
            raw = read_json(config)
            raw.update(model="changed-model", prompt_file="missing-prompt", reference_image_dir="missing-references")
            raw["dataset"]["image_dir"] = "missing-new-dataset"
            write_json(config, raw)
            output = io.StringIO()
            with patch("industrial_fewshot.runner.run", side_effect=AssertionError("no inference")), \
                 patch("industrial_fewshot.runner.load_config", side_effect=AssertionError("no model config")), \
                 contextlib.redirect_stdout(output):
                code = pipeline.main(["--config", str(config), "--evaluate-only", str(run_dir), '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["metrics"]["mAP50"], 1)

    def test_default_cli_prints_tables_and_persists_text_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = saved_run(Path(directory))
            original_records = (run_dir / 'records.jsonl').read_bytes()
            output = io.StringIO()
            with patch('industrial_fewshot.runner.run', side_effect=AssertionError('no inference')), contextlib.redirect_stdout(output):
                code = pipeline.main(['--evaluate-only', str(run_dir)])
            self.assertEqual(code, 0)
            self.assertIn('Average Precision', output.getvalue())
            self.assertIn('VOC-style Evaluation', output.getvalue())
            self.assertNotIn('"annotations"', output.getvalue())
            self.assertEqual((run_dir / 'evaluation.txt').read_text(encoding='utf-8').strip(), output.getvalue().strip())
            self.assertEqual((run_dir / 'records.jsonl').read_bytes(), original_records)

    def test_incomplete_run_is_rejected_instead_of_hiding_unprocessed_images(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = saved_run(Path(directory))
            metadata = read_json(run_dir / "run.json")
            metadata["image_count"] = 100
            write_json(run_dir / "run.json", metadata)
            with self.assertRaisesRegex(ValueError, "推理记录不完整"):
                evaluate_saved_run(run_dir)
            self.assertFalse((run_dir / "metrics.json").exists())

    def test_missing_confidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = saved_run(Path(directory))
            path = run_dir / "records.jsonl"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["objects"][0].pop("confidence")
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺少 confidence"):
                evaluate_saved_run(run_dir)

    def test_preflight_missing_dependency_stops_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            with patch("industrial_fewshot.runner.check_evaluation_dependencies", side_effect=RuntimeError("missing numpy")):
                with self.assertRaisesRegex(RuntimeError, "missing numpy"):
                    import asyncio
                    asyncio.run(pipeline.run(pipeline.load_config(config), client_factory=lambda **_: self.fail("must not infer")))
            self.assertFalse((root / "results").exists())

    def test_evaluation_failure_keeps_completed_inference_summary(self):
        import asyncio
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            client = FakeClient([response()])
            with patch("industrial_fewshot.runner.evaluate_records", side_effect=RuntimeError("evaluation interrupted")):
                with self.assertRaisesRegex(RuntimeError, "无需重跑推理"):
                    asyncio.run(pipeline.run(pipeline.load_config(config), client_factory=lambda **_: client))
            run_dir = next((root / "results").glob("*/run.json")).parent
            summary = read_json(run_dir / "summary.json")
            self.assertEqual(summary["evaluation_status"], "failed")
            self.assertEqual(summary["success_request_count"], 1)
            self.assertEqual(len(client.requests), 1)
            before = (run_dir / "records.jsonl").read_bytes()
            report = evaluate_saved_run(run_dir)
            self.assertEqual(report["metrics"]["mAP50"], 1)
            self.assertEqual((run_dir / "records.jsonl").read_bytes(), before)

    def test_dry_run_does_not_write_derived_results(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = saved_run(Path(directory))
            report = evaluate_saved_run(run_dir, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertFalse((run_dir / "metrics.json").exists())
            self.assertFalse((run_dir / "summary.json").exists())

    def test_existing_failed_request_remains_counted_as_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = saved_run(Path(directory))
            record = {"target_id": "sample", "relative_path": "frame.jpg", "status": "failed", "error": "offline"}
            (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = evaluate_saved_run(run_dir)
            self.assertEqual(report["metrics"]["fn"], 1)
            self.assertEqual(report["evaluation_status"], "incomplete")
