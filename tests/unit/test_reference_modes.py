import tempfile
import unittest
from pathlib import Path

from PIL import Image

import industrial_fewshot.prompts.references as references
import industrial_fewshot.config as scene_config
from industrial_fewshot.data.labelme import read_json, write_json
from tests.support import fixture, make_image, make_reference_pair, reference_config


class ReferenceModesTest(unittest.TestCase):
    def prepare(self, root, mode):
        config = reference_config(root)
        directory = Path(config["reference_image_dir"])
        make_reference_pair(directory, "positive")
        make_image(directory / "positive_crop_01_target.png", (40, 80))
        make_image(directory / "negative.jpg")
        references.configure_reference_inputs(config, mode)
        references.prepare_reference_examples(config)
        return config

    def test_four_modes_send_only_selected_reference_inputs_and_current_image(self):
        for mode, json_count, crop_count in (("text_only", 0, 0), ("text_json", 1, 0),
                                            ("text_crops", 0, 1), ("text_json_crops", 1, 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = self.prepare(Path(directory), mode)
                content = references.build_messages(config, "SHARED_PROMPT", config["reference_examples"], "CURRENT_FULL")
                urls = [b["image_url"]["url"] for b in content if b["type"] == "image_url"]
                self.assertEqual(len(urls), 1 + crop_count)
                self.assertEqual(urls[-1], "CURRENT_FULL")
                self.assertEqual(len(config["reference_examples"]), json_count)
                self.assertEqual(config["reference_negatives"], [])
                text = "".join(b["text"] for b in content if b["type"] == "text")
                self.assertEqual('"imageWidth":640' in text, bool(json_count))
                self.assertIn("SHARED_PROMPT", text)
                self.assertNotIn("红框", text)

    def test_json_only_needs_no_reference_image_and_strips_embedded_image_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = reference_config(root, max_reference_crops=0, max_reference_negatives=0)
            image, path = make_reference_pair(Path(config["reference_image_dir"]), "positive")
            image.write_bytes(b"NOT_AN_IMAGE")
            document = read_json(path)
            document["imageData"] = "EMBEDDED_IMAGE_SENTINEL"
            document["imagePath"] = "missing.jpg"
            write_json(path, document)
            references.configure_reference_inputs(config, "text_json")
            references.prepare_reference_examples(config)
            text = config["reference_examples"][0]["compact_json"]
            self.assertNotIn("EMBEDDED_IMAGE_SENTINEL", text)
            self.assertNotIn("rendered_data_url", config["reference_examples"][0])

    def test_only_active_input_content_changes_reference_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare(root, "text_json")
            before = config["reference_manifest_sha256"]
            ref = root / "references"
            Image.new("RGB", (640, 360), "red").save(ref / "positive.jpg")
            Image.new("RGB", (40, 80), "red").save(ref / "positive_crop_01_target.png")
            references.prepare_reference_examples(config)
            self.assertEqual(before, config["reference_manifest_sha256"])
            document = read_json(ref / "positive.json")
            document["imageWidth"] = 1000
            write_json(ref / "positive.json", document)
            references.prepare_reference_examples(config)
            self.assertNotEqual(before, config["reference_manifest_sha256"])

    def test_mode_switch_restores_original_sample_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            config = reference_config(Path(directory), max_reference_examples=3, max_reference_crops=2)
            references.configure_reference_inputs(config, "text_only")
            references.configure_reference_inputs(config, "text_json_crops")
            self.assertEqual(config["max_reference_examples"], 3)
            self.assertEqual(config["max_reference_crops"], 2)
            self.assertEqual(config["max_reference_negatives"], 0)

    def test_required_reference_type_is_not_silently_omitted(self):
        for mode in ("text_json", "text_crops", "text_json_crops"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = reference_config(Path(directory))
                references.configure_reference_inputs(config, mode)
                with self.assertRaises(ValueError):
                    references.prepare_reference_examples(config)

    def test_text_only_scene_needs_no_reference_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            raw.pop("reference_image_dir")
            raw["reference_mode"] = "text_only"
            write_json(path, raw)
            config = scene_config.load_config(path)["targets"][0]["config"]
            references.prepare_reference_examples(config)
            self.assertEqual(config["reference_examples"], [])

    def test_invalid_modes_and_ablation_settings_fail_validation(self):
        cases = [{"reference_mode": "bad"}, {"ablation": True}, {"ablation": {"enabled": "yes"}},
                 {"ablation": {"modes": []}}, {"ablation": {"modes": ["text_only", "text_only"]}},
                 {"ablation": {"modes": ["configured"]}}, {"ablation": {"modes": [None]}}]
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                path, raw = fixture(Path(directory))
                raw.update(changes)
                write_json(path, raw)
                with self.assertRaises(ValueError):
                    scene_config.load_config(path)
