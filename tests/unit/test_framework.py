import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util

from PIL import Image

from industrial_fewshot.configuration import load_document, redact
from industrial_fewshot.data.preprocessing import assign_splits, to_coco
from industrial_fewshot.detectors.base import ImageInput
from industrial_fewshot.detectors.groundingdino_text import groundingdino_settings, GroundingDINORunner
from industrial_fewshot.evaluation.fewshot_stability import summarize
from industrial_fewshot.postprocess.boxes import nms
import industrial_fewshot.paths as locations


class FrameworkUnitTest(unittest.TestCase):
    def test_visual_experiments_preserve_shared_settings_and_explicit_shot_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'configs'
            project_configs = Path(__file__).resolve().parents[2] / 'configs'
            # Compose actual entrypoints against an edited common JSON.
            for name in ('common/diaowu.yaml', 'common/diaowu_yoloe_config.json',
                         *(f'specific/visual_{count}shot.yaml' for count in (1, 5, 10, 25))):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project_configs / name, destination)
            legacy = root / 'common/diaowu_yoloe_config.json'
            raw = json.loads(legacy.read_text(encoding='utf-8-sig'))
            raw['evaluation']['score_threshold'] = 0.73
            raw['yoloe'].update(confidence=0.27, selection_seed=4, positive_count=50)
            legacy.write_text(json.dumps(raw), encoding='utf-8')
            for count in (1, 5, 10, 25):
                with self.subTest(shot=count):
                    merged, _ = load_document(root / f'specific/visual_{count}shot.yaml')
                    self.assertEqual(merged['evaluation']['score_threshold'], 0.73)
                    self.assertEqual(merged['yoloe']['confidence'], 0.27)
                    self.assertEqual(merged['yoloe']['selection_seed'], 4)
                    self.assertEqual(merged['yoloe']['positive_count'], count)
                    self.assertIn('manifest_file', merged['dataset'])

    def test_yaml_inheritance_anchors_paths_in_declaring_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            (root / "models/base.yaml").write_text("prompt_file: prompt.txt\ninference: {concurrency: 3, max_tokens: 20}\nlabels: [one]\n", encoding="utf-8")
            path = root / "experiment.yaml"
            path.write_text("extends: models/base.yaml\ninference: {max_tokens: 100}\nlabels: [two]\n", encoding="utf-8")
            raw, sources = load_document(path)
            self.assertEqual(raw["prompt_file"], str(root / "models/prompt.txt"))
            self.assertEqual(raw["inference"], {"concurrency": 3, "max_tokens": 100})
            self.assertEqual(raw["labels"], ["two"])
            self.assertEqual(len(sources), 2)

    def test_cyclic_extends_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("extends: b.yaml", encoding="utf-8")
            (root / "b.yaml").write_text("extends: a.yaml", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_document(root / "a.yaml")

    def test_redaction_preserves_inference_settings(self):
        raw = {"model": {"api_key": "SECRET", "name": "model"}, "inference": {"max_tokens": 5}}
        result = redact(raw)
        self.assertNotIn("SECRET", str(result))
        self.assertEqual(result["inference"]["max_tokens"], 5)
        self.assertEqual(raw["model"]["api_key"], "SECRET")

    def test_renamed_project_resolves_old_paths_and_preserves_reference_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            old, new = parent / "annotationtool", parent / "industrial_fewshot_detection"
            current = new / "data/references/diaowu/a.png"
            current.parent.mkdir(parents=True)
            current.write_bytes(b"unchanged")
            with patch.object(locations, "PROJECT_ROOT", new), patch.object(locations, "CACHE_ORIGIN_ROOT", old):
                self.assertEqual(locations.resolve_location(old / "data/DiaoWuexample_prepared/a.png"), current)
                identity = locations.stable_reference_identity({"image_path": str(current)})
                self.assertEqual(identity["image_path"], str(old / "data/DiaoWuexample_prepared/a.png"))

    def test_video_groups_are_disjoint_and_seed_reproducible(self):
        samples = [{"relative_path": f"video{video}/frame{frame}.jpg"} for video in range(6) for frame in range(3)]
        ratios = {"train": .5, "val": 1/6, "test": 1/3}
        splits = assign_splits(samples, ratios, 4, "parent")
        groups = [{str(Path(row["relative_path"]).parent) for row in rows} for rows in splits.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(assign_splits(samples, ratios, 4, "parent"), splits)
        self.assertEqual(sum(map(len, splits.values())), len(samples))

    def test_one_video_cannot_be_split_across_train_val_test(self):
        with self.assertRaises(ValueError):
            assign_splits([{"relative_path": "frame.jpg"}], {"train": .6, "val": .2, "test": .2}, 0, "parent")

    def test_coco_keeps_known_negatives_but_excludes_unknown_truth(self):
        def sample(name, truth):
            return {"relative_path": name, "ground_truth": truth, "width": 100, "height": 50, "image_sha256": "hash"}
        coco = to_coco([sample("negative.jpg", []), sample("unknown.jpg", None),
                        sample("positive.jpg", [{"label": "target", "bbox_pixels_xyxy": [1.5, 2, 21.5, 30]}])], ["target"], Path("images"))
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(coco["annotations"][0]["bbox"], [1.5, 2, 20, 28])
        self.assertEqual(coco["info"]["unknown_truth_excluded"], 1)

    def test_detector_input_excludes_test_truth(self):
        item = {"path": Path("image.jpg"), "relative_path": "image.jpg", "width": 100, "height": 50,
                "image_sha256": "hash", "ground_truth": "TEST_TRUTH_SECRET", "ground_truth_source": "json"}
        self.assertNotIn("ground_truth", ImageInput.from_item(item).as_item())
        self.assertNotIn("TEST_TRUTH_SECRET", str(ImageInput.from_item(item)))

    def test_class_aware_nms_keeps_other_class_and_float_boxes(self):
        objects = [{"label": label, "bbox_pixels_xyxy": [1.25, 2, 20.25, 30], "confidence": confidence}
                   for label, confidence in [("a", .9), ("a", .8), ("b", .7)]]
        result = nms(objects, .5, 10)
        self.assertEqual([o["label"] for o in result], ["a", "b"])
        self.assertEqual(result[0]["bbox_pixels_xyxy"][0], 1.25)

    def test_grounding_settings_validate_thresholds_and_unknown_keys(self):
        for settings in ({"box_threshold": float("nan")}, {"max_detections": True}, {"negative_count": 5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                groundingdino_settings(settings, Path.cwd())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for mocked tensor output")
    def test_grounding_postprocess_uses_height_width_and_pixel_boxes(self):
        import torch
        class Inputs(dict):
            input_ids = torch.tensor([[1]])
            def to(self, device):
                return self
        class Processor:
            def __call__(self, **kwargs):
                self.input = kwargs
                return Inputs()
            def post_process_grounded_object_detection(self, *args, **kwargs):
                self.post = kwargs
                return [{"boxes": torch.tensor([[1.25, 2.5, 80.5, 40.25]]), "scores": torch.tensor([.9])}]
        runner = GroundingDINORunner.__new__(GroundingDINORunner)
        runner.torch, runner.processor, runner.model = torch, Processor(), lambda **kwargs: SimpleNamespace()
        runner.settings = groundingdino_settings({}, Path.cwd())
        runner.config = {"labels": [{"name": "target"}]}
        buffer = io.BytesIO()
        Image.new("RGB", (100, 50)).save(buffer, format="PNG")
        result = runner.predict(buffer.getvalue())
        self.assertEqual(runner.processor.post["target_sizes"], [(50, 100)])
        self.assertEqual(result["objects"][0]["bbox_pixels_xyxy"], [1.25, 2.5, 80.5, 40.25])
        self.assertEqual(set(runner.processor.input), {"images", "text", "return_tensors"})

    def test_stability_single_seed_std_is_undefined_and_duplicates_rejected(self):
        row = {"backend": "yoloe", "model": "m", "reference_mode": "O+J", "positive_original_count": 5,
               "negative_original_count": 0, "selection_seed": 0, "mAP50": .5, "failed_request_count": 1}
        report = {"dataset_sha256": "same", "evaluation": {}, "missing_ground_truth": "negative", "rows": [row]}
        self.assertIsNone(summarize([report])["rows"][0]["mAP50"]["sample_std"])
        with self.assertRaises(ValueError):
            summarize([report, report])
        other = copy.deepcopy(report)
        other["rows"][0].update(selection_seed=1, mAP50=.7)
        stats = summarize([report, other])["rows"][0]
        self.assertAlmostEqual(stats["mAP50"]["mean"], .6)
        self.assertEqual(stats["failed_request_count"], 2)
