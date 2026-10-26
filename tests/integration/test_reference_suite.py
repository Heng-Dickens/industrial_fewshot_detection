import asyncio
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.runner import load_config
from industrial_fewshot.data.supports import DEFAULT_VARIANTS, ReferencePool, group_specs, suite_settings
from industrial_fewshot.prompts.references import build_messages
from tests.support import FakeClient, fixture, response
from industrial_fewshot.data.prepare import prepare_dataset


def suite_fixture(root, **settings):
    path, raw = fixture(root)
    source = root / "source"
    source.mkdir()
    for index in range(25):
        Image.new("RGB", (64, 48), (index * 8, 40, 80)).save(source / f"p{index:02d}.png")
        document = {"imagePath": f"p{index:02d}.png", "imageWidth": 64, "imageHeight": 48,
                    "flags": {"REFERENCE_JSON_SENTINEL": True},
                    "shapes": [{"label": "target", "shape_type": "rectangle", "points": [[2, 3], [20, 30]]}]}
        if index == 0:
            document["shapes"].append({"label": "target", "shape_type": "rectangle", "points": [[30, 3], [50, 30]]})
        write_json(source / f"p{index:02d}.json", document)
        Image.new("RGB", (64, 48), (10, index * 8, 200)).save(source / f"n{index:02d}.png")
    prepare_dataset(source, root / "prepared")
    truth = read_json(root / "images/frame.json")
    truth["flags"] = {"TEST_TRUTH_SECRET": True}
    write_json(root / "images/frame.json", truth)
    raw.update(reference_image_dir="prepared", reference_mode="text_only",
               experiment_suite={"enabled": True, "positive_counts": [1, 5, 10, 25], **settings})
    write_json(path, raw)
    return path, raw


class ReferenceSuiteTest(unittest.TestCase):
    def execute(self, path, dry_run=False, fail=False):
        clients = []
        def factory(**_):
            if fail:
                self.fail("Repeated suite must reuse cached predictions")
            client = FakeClient([response(), response()])
            clients.append(client)
            return client
        report = asyncio.run(run_ablation(load_config(path), dry_run=dry_run, client_factory=factory))
        return report, clients

    def test_settings_and_57_unique_groups_without_original_only(self):
        settings = suite_settings({"enabled": True, "positive_counts": [1, 5, 10, 25]})
        specs = group_specs(settings)
        self.assertEqual(len(specs), 57)
        self.assertEqual(len({s["id"] for s in specs}), 57)
        self.assertEqual(sum(s["positive_count"] == 0 for s in specs), 1)
        self.assertEqual({s["variant"] for s in specs[1:]}, set(DEFAULT_VARIANTS))
        self.assertNotIn("O", DEFAULT_VARIANTS)
        self.assertIn("O+C+B", DEFAULT_VARIANTS)
        for spec in specs[1:]:
            self.assertIn(spec["positive_count"], [1, 5, 10, 25])
            self.assertIn(spec["negative_count"], [0, spec["positive_count"]])

    def test_defaults_use_five_positives_and_five_negatives_in_15_groups(self):
        settings = suite_settings({"enabled": True})
        self.assertEqual(settings["positive_counts"], [5])
        groups = group_specs(settings)
        self.assertEqual(len(groups), 15)
        self.assertTrue(all(s["positive_count"] == 5 for s in groups[1:]))
        self.assertEqual(sum(s["negative_count"] == 5 for s in groups), 7)

    def test_invalid_settings_fail_clearly(self):
        for settings in ({"positive_counts": [True]}, {"positive_counts": [0]},
                         {"positive_counts": [1, 1]}, {"positive_variants": ["O"]},
                         {"positive_variants": ["O+C", "O+C"]}, {"negative_variants": [1]},
                         {"negative_variants": [True, True]}, {"selection_seed": False},
                         {"enabled": "true"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                suite_settings(settings)

    def test_all_groups_nested_same_pairs_counts_payloads_and_no_truth_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = suite_fixture(root)
            report, clients = self.execute(path, dry_run=True)
            self.assertFalse(clients)
            self.assertEqual(report["group_count"], 57)
            self.assertEqual(report["scheduled_inference_count"], 57)
            self.assertFalse((root / "results").exists())
            prefixes = {}
            config = load_config(path)
            pool = ReferencePool(config["targets"][0]["config"], 0)
            for group in report["groups"][1:]:
                meta = group["experiment"]
                selection = meta["reference_sets"][0]
                k, n = meta["positive_count"], meta["negative_count"]
                self.assertEqual(len(selection["positive_sample_ids"]), k)
                self.assertEqual(len(selection["negative_sample_ids"]), n)
                self.assertEqual(meta["source_reference_count"], k + n)
                if k in prefixes:
                    self.assertEqual(prefixes[k], selection["positive_sample_ids"])
                prefixes[k] = selection["positive_sample_ids"]
                prepared = copy.deepcopy(config["targets"][0]["config"])
                pool.prepare(prepared, meta)
                content = build_messages(prepared, "PROMPT", [], "CURRENT_IMAGE")
                text = str(content)
                self.assertNotIn("TEST_TRUTH_SECRET", text)
                self.assertEqual("REFERENCE_JSON_SENTINEL" in text, "json" in meta["positive_inputs"])
                self.assertEqual("负样本" in text, n > 0)
                self.assertEqual(sum(b["type"] == "image_url" for b in content), meta["reference_image_count"] + 1)
                self.assertEqual(content[-1]["image_url"]["url"], "CURRENT_IMAGE")
                for sample in prepared["reference_samples"]:
                    stem = Path(sample["sample_id"]).stem
                    self.assertEqual(Path(sample["original"]["image_path"]).stem, stem)
                    for crop in sample.get("crops", []):
                        self.assertTrue(Path(crop["image_path"]).stem.startswith(stem + "_crop_"))
                    if "boxed" in sample:
                        self.assertEqual(Path(sample["boxed"]["image_path"]).stem, stem + "_boxed")
                        self.assertNotEqual(sample["original"]["rendered_data_url"], sample["boxed"]["rendered_data_url"])
                    if "json" in sample:
                        self.assertEqual(Path(sample["json"]["json_path"]).stem, stem)
            for small, large in ((1, 5), (5, 10), (10, 25)):
                self.assertEqual(prefixes[small], prefixes[large][:small])
            full = next(g for g in report["groups"] if g["experiment"]["id"] == "O_C_B_J_k25_posneg")
            self.assertEqual(full["experiment"]["reference_image_count"], 101)
            self.assertEqual(full["experiment"]["source_reference_count"], 50)

    def test_full_fake_run_writes_57_rows_and_repeat_reuses_all_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = suite_fixture(Path(directory))
            first, clients = self.execute(path)
            self.assertEqual(first["executed_inference_count"], 57)
            self.assertEqual(len(clients), 57)
            self.assertEqual(len(first["rows"]), 57)
            self.assertEqual(first["status"], "complete")
            self.assertTrue(all(r["image_accuracy"] == 1 for r in first["rows"]))
            self.assertTrue(all(r["delta_mAP50"] == 0 for r in first["rows"]))
            self.assertEqual(first["rows"][-1]["reference_image_count"], 101)
            repeated, clients = self.execute(path, fail=True)
            self.assertFalse(clients)
            self.assertEqual(repeated["reused_prediction_count"], 57)
            self.assertEqual(repeated["executed_inference_count"], 0)
            self.assertEqual(first["dataset_sha256"], repeated["dataset_sha256"])

    def test_unchanged_unused_materials_do_not_invalidate_other_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = suite_fixture(root, positive_counts=[25], positive_variants=["O+C+B", "O+C+B+J"])
            self.execute(path)
            document_path = root / "prepared/positive/annotations/p00.json"
            document = read_json(document_path)
            document["flags"]["change"] = True
            write_json(document_path, document)
            updated, _ = self.execute(path)
            self.assertEqual([g["executed_inference_count"] for g in updated["groups"]], [0, 0, 0, 1, 1])
            # Negative image changes affect only groups that actually send negatives.
            Image.new("RGB", (64, 48), "purple").save(root / "prepared/negative/images/n00.png")
            changed, _ = self.execute(path)
            self.assertEqual([g["executed_inference_count"] for g in changed["groups"]], [0, 0, 1, 0, 1])

    def test_reference_overlap_is_excluded_for_every_group_including_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = suite_fixture(root, positive_counts=[1], positive_variants=["O+C+B"])
            shutil.copyfile(root / "prepared/positive/images/p00.png", root / "images/renamed.png")
            raw["dataset"]["missing_ground_truth"] = "negative"
            write_json(path, raw)
            report, _ = self.execute(path, dry_run=True)
            self.assertEqual(report["excluded_reference_overlap"], ["renamed.png"])
            self.assertTrue(all(g["image_paths"] == ["frame.jpg"] for g in report["groups"]))
            raw["experiment_suite"]["exclude_reference_overlap"] = False
            write_json(path, raw)
            with self.assertRaisesRegex(ValueError, "内容重复"):
                self.execute(path, dry_run=True)

    def test_missing_assets_and_insufficient_counts_fail_before_any_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = suite_fixture(root, positive_counts=[26])
            with self.assertRaisesRegex(ValueError, "参考数量不足"):
                self.execute(path)
            self.assertFalse((root / "results").exists())
            raw["experiment_suite"]["positive_counts"] = [25]
            write_json(path, raw)
            (root / "prepared/positive/boxed_images/p00_boxed.png").unlink()
            with self.assertRaisesRegex(ValueError, "参考资源缺失"):
                self.execute(path)
            self.assertFalse((root / "results").exists())

    def test_review_uses_same_paired_inputs_without_test_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = suite_fixture(Path(directory), positive_counts=[1], positive_variants=["O+C+B+J"], negative_variants=[True])
            raw["inference"]["enable_completeness_review"] = True
            write_json(path, raw)
            _, clients = self.execute(path)
            self.assertEqual(len(clients), 2)
            for client in clients:
                self.assertEqual(len(client.requests), 2)
                for request in client.requests:
                    self.assertNotIn("TEST_TRUTH_SECRET", str(request["messages"]))
            for request in clients[1].requests:
                self.assertIn("REFERENCE_JSON_SENTINEL", str(request["messages"]))
                self.assertIn("负样本", str(request["messages"]))
