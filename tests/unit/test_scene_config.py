import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import industrial_fewshot.config as scene_config
from industrial_fewshot.data.labelme import write_json
from tests.support import FakeClient, fixture, response


class SceneConfigTest(unittest.TestCase):
    def test_unspecified_null_and_empty_output_use_input_sibling(self):
        for value in (None, "", "   ", "missing"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root)
                if value == "missing":
                    raw.pop("output_dir")
                else:
                    raw["output_dir"] = value
                write_json(path, raw)
                config = scene_config.load_config(path)
                self.assertEqual(config["output_dir"], (root / "images_annotations").resolve())
                self.assertFalse(config["output_dir"].exists())

    def test_custom_suffix_follows_overridden_image_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw.update(output_dir=None, output_suffix="_预测")
            write_json(path, raw)
            source = root / "车间/00000"
            source.mkdir(parents=True)
            config = scene_config.load_config(path, image_dir=source)
            self.assertEqual(config["output_dir"], (root / "车间/00000_预测").resolve())

    def test_explicit_config_and_command_line_output_take_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            self.assertEqual(scene_config.load_config(path)["output_dir"], (root / "results").resolve())
            chosen = root / "chosen"
            self.assertEqual(scene_config.load_config(path, output_dir=chosen)["output_dir"], chosen.resolve())
            with self.assertRaisesRegex(ValueError, "output_dir"):
                scene_config.load_config(path, output_dir=root / "images/predictions")

    def test_suffix_cannot_escape_sibling_or_make_invalid_folder_name(self):
        for suffix in ("", " ", None, 1, "../results", "_a/b", "_a\\b", "_a:b", "_a*", "_a\n", "_a.", "_a "):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path, raw = fixture(Path(directory))
                raw.update(output_dir=None, output_suffix=suffix)
                write_json(path, raw)
                with self.assertRaisesRegex(ValueError, "output_suffix"):
                    scene_config.load_config(path)

    def test_default_output_still_protects_separate_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw.pop("output_dir")
            raw["dataset"]["ground_truth_dir"] = "images_annotations"
            (root / "images_annotations").mkdir()
            write_json(path, raw)
            with self.assertRaisesRegex(ValueError, "output_dir"):
                scene_config.load_config(path)

    def test_scene_validation_rejects_bad_backend_labels_and_reference_limits(self):
        cases = [
            {"backend": "unknown"}, {"labels": []}, {"max_reference_crops": -1},
            {"max_reference_examples": True}, {"max_reference_negatives": -1},
        ]
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                path, raw = fixture(Path(directory))
                raw.update(changes)
                write_json(path, raw)
                with self.assertRaises(ValueError):
                    scene_config.load_config(path)

    def test_all_strategies_share_crop_and_negative_support(self):
        for backend in ("generic", "diaowu", "diaogou"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                path, raw = fixture(Path(directory), backend=backend)
                raw.update(max_reference_crops=2, max_reference_negatives=3)
                write_json(path, raw)
                config = scene_config.load_config(path)["targets"][0]["config"]
                self.assertEqual(config["max_reference_crops"], 2)
                self.assertEqual(config["max_reference_negatives"], 3)

    def test_output_cannot_be_inside_protected_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            raw["output_dir"] = "images/predictions"
            write_json(path, raw)
            with self.assertRaisesRegex(ValueError, "output_dir"):
                scene_config.load_config(path)

    def test_flat_scene_config_loads_legacy_model_labels_and_generation_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw.update(raw.pop("dataset"))
            raw.update(raw.pop("inference"))
            raw.update(model="flat-scene-model", base_url="http://localhost:8080/v1", temperature=0.25)
            write_json(path, raw)
            config = scene_config.load_config(path)
            self.assertEqual(config["image_dir"], (root / "images").resolve())
            self.assertEqual(config["targets"][0]["config"]["model"], "flat-scene-model")
            self.assertEqual(config["targets"][0]["config"]["temperature"], 0.25)

    def test_scene_paths_are_relative_to_selected_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            location = root / "arbitrary/config/location"
            location.mkdir(parents=True)
            for key in ("reference_image_dir", "prompt_file", "review_prompt_file", "output_dir"):
                raw[key] = "../../../" + raw[key]
            raw["dataset"]["image_dir"] = "../../../images"
            moved = location / "any_scene_name.json"
            write_json(moved, raw)
            config = scene_config.load_config(moved)
            self.assertEqual(config["image_dir"], (root / "images").resolve())
            self.assertEqual(config["targets"][0]["prompt_file"], str((root / "prompt.txt").resolve()))
