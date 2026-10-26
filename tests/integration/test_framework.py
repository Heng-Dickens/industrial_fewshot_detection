import asyncio
import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.integration.test_yoloe_backend import local_fixture, FakeRunner
from tests.support import FakeClient, response
from industrial_fewshot.config import load_config
from industrial_fewshot.data.preprocessing import prepare_experiment_dataset
from industrial_fewshot.data.dataset import inspect_dataset
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.runner import run
from industrial_fewshot.evaluation.compare import compare_runs
from industrial_fewshot.evaluation.visualize import render_run


class FrameworkIntegrationTest(unittest.TestCase):
    def test_prepared_cohort_supports_and_frozen_image_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = local_fixture(root)
            raw.update(preparation={"prepared_dir": "processed", "splits_dir": "splits", "supports_dir": "supports",
                                    "support_counts": [1, 5], "support_seeds": [0]}, support_manifest_dir="supports")
            raw["dataset"]["manifest_file"] = "splits/test.json"
            write_json(path, raw)
            report = prepare_experiment_dataset(load_config(path))
            self.assertEqual(report["image_count"], 1)
            one = read_json(root / "supports/1shot/seed_0.json")
            five = read_json(root / "supports/5shot/seed_0.json")
            self.assertEqual(one["positive_samples"], five["positive_samples"][:1])
            planned = asyncio.run(run(load_config(path), dry_run=True))
            self.assertEqual(planned["support_manifests"][0]["positive_count"], 5)
            five["positive_samples"].reverse()
            write_json(root / "supports/5shot/seed_0.json", five)
            with self.assertRaisesRegex(ValueError, "Support selection"):
                asyncio.run(run(load_config(path), dry_run=True))
            prepare_experiment_dataset(load_config(path))
            Image.new("RGB", (1000, 500), "red").save(root / "images/frame.jpg")
            with self.assertRaises(ValueError):
                asyncio.run(run(load_config(path), dry_run=True))

    def test_three_backends_compare_same_cohort_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = local_fixture(root)
            first = asyncio.run(run(load_config(path), local_runner_factory=FakeRunner))
            raw.update(backend="groundingdino", output_dir="ovd_results", groundingdino={"checkpoint": "absent"})
            write_json(path, raw)
            second = asyncio.run(run(load_config(path), local_runner_factory=FakeRunner))
            raw.update(backend="generic", output_dir="mllm_results", reference_mode="text_only")
            write_json(path, raw)
            third = asyncio.run(run(load_config(path), client_factory=lambda **_: FakeClient([response()])))
            result = compare_runs([Path(r["run_dir"]) for r in (first, second, third)], root / "comparison")
            self.assertEqual([r["method"] for r in result["rows"]], ["YOLOE", "GroundingDINO", "VLM"])
            self.assertTrue(all(r["mAP50"] == 1 for r in result["rows"]))
            snapshot = read_json(Path(third["run_dir"]) / "config.resolved.json")
            self.assertEqual(snapshot["model"]["api_key"], "<redacted>")
            visualization = render_run(Path(second["run_dir"]), root / "visualization")
            self.assertTrue(Path(visualization["images"][0]["output"]).is_file())
