import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import industrial_fewshot.prompts.references as target
from tests.support import LABEL, make_image, make_reference_pair, reference_config


class DiscoverReferencePairsTest(unittest.TestCase):
    def test_pairs_images_sorted_and_capped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_reference_pair(root, "b")
            make_reference_pair(root, "a")
            orphan = root / "c.json"
            orphan.write_text(
                json.dumps(
                    {
                        "version": "6.3.1",
                        "flags": {},
                        "shapes": [],
                        "imagePath": "ghost.jpg",
                        "imageData": None,
                        "imageHeight": 360,
                        "imageWidth": 640,
                    }
                ),
                encoding="utf-8",
            )
            (root / "notes.json").write_text(
                json.dumps({"not": "labelme"}), encoding="utf-8"
            )
            pairs = target.discover_reference_pairs(root, 2)
            self.assertEqual([p["json_path"].name for p in pairs], ["a.json", "b.json"])
            for pair in pairs:
                self.assertTrue(pair["image_path"].is_file())
                self.assertEqual(len(pair["image_sha256"]), 64)

    def test_falls_back_to_same_stem_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "fallback.jpg")
            (root / "fallback.json").write_text(
                json.dumps(
                    {
                        "version": "6.3.1",
                        "flags": {},
                        "shapes": [],
                        "imagePath": "",
                        "imageData": None,
                        "imageHeight": 360,
                        "imageWidth": 640,
                    }
                ),
                encoding="utf-8",
            )
            pairs = target.discover_reference_pairs(root, 6)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["image_path"].name, "fallback.jpg")



class ReferenceRenderingTest(unittest.TestCase):
    def test_render_reference_image_returns_jpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            image_path, json_path = make_reference_pair(root, "sample")
            document = json.loads(json_path.read_text(encoding="utf-8"))
            data, mime = target.render_reference_image(
                image_path, document["shapes"]
            )
            self.assertTrue(data.startswith(b"\xff\xd8"))
            self.assertEqual(mime, "image/jpeg")

    def test_compact_document_rounds_points_and_drops_image_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            _, json_path = make_reference_pair(root, "sample")
            document = json.loads(json_path.read_text(encoding="utf-8"))
            compact = target.compact_labelme_document(document)
            payload = json.loads(compact)
            self.assertNotIn("imageData", payload)
            self.assertEqual(payload["version"], "6.3.1")
            self.assertEqual(payload["shapes"][0]["points"], [[10.0, 20.0], [300.0, 320.0]])



class NegativeReferenceTest(unittest.TestCase):
    def test_discovers_images_without_annotation(self):
        # 参考目录里没有同名 .json 的图片 = 不符合条件的负例
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            ref = root / "references"
            ref.mkdir()
            make_reference_pair(ref, "positive")
            make_image(ref / "negative.jpg")
            negatives = target.discover_negative_references(ref, 6)
            self.assertEqual(
                [Path(item["image_path"]).name for item in negatives], ["negative.jpg"]
            )

    def test_zero_cap_disables_negatives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            ref = root / "references"
            ref.mkdir()
            make_image(ref / "negative.jpg")
            self.assertEqual(target.discover_negative_references(ref, 0), [])

    def test_prepare_collects_negatives_and_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            config = reference_config(root)
            ref = Path(config["reference_image_dir"])
            make_reference_pair(ref, "positive")
            make_image(ref / "negative.jpg")
            target.prepare_reference_examples(config)
            self.assertEqual(len(config["reference_examples"]), 1)
            self.assertEqual(len(config["reference_negatives"]), 1)
            first = config["reference_manifest_sha256"]
            # 负例增删也必须让检查点失效
            make_image(ref / "negative2.jpg")
            target.prepare_reference_examples(config)
            self.assertNotEqual(first, config["reference_manifest_sha256"])

    def test_build_messages_sends_positives_then_negatives_then_target(self):
        config = {
            "labels": [{"name": LABEL, "description": "吊物"}],
            "extra_instruction": "",
            "reference_negatives": [{"rendered_data_url": "data:image/jpeg;base64,NEG"}],
        }
        examples = [
            {"rendered_data_url": "data:image/jpeg;base64,POS", "compact_json": "{}"}
        ]
        content = target.build_messages(
            config, "PROMPT", examples, "data:image/jpeg;base64,CUR"
        )
        urls = [
            block["image_url"]["url"]
            for block in content
            if block.get("type") == "image_url"
        ]
        self.assertEqual(
            urls,
            [
                "data:image/jpeg;base64,POS",
                "data:image/jpeg;base64,NEG",
                "data:image/jpeg;base64,CUR",
            ],
        )
        texts = [block["text"] for block in content if block.get("type") == "text"]
        self.assertTrue(any("未标注示例" in text for text in texts))
        self.assertTrue(any("正确答案是不标注任何框" in text for text in texts))



class CropReferenceTest(unittest.TestCase):
    def test_shared_crop_messages_do_not_inject_hanging_object_rules(self):
        config = {"labels": [{"name": "forklift"}], "reference_crops": [{"rendered_data_url": "CROP"}]}
        content = target.build_messages(config, "FORKLIFT_RULES", [], "FULL")
        text = "".join(b["text"] for b in content if b["type"] == "text")
        self.assertIn("FORKLIFT_RULES", text)
        self.assertNotIn("吊钩", text)
        self.assertNotIn("吊物", text)

    def test_discovers_sorted_crops_and_never_uses_them_as_negatives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("b_crop_01_hanging_object.PNG", "a_crop_02_hanging_object.jpg", "negative.jpg"):
                make_image(root / name)
            (root / "not_crop_01_object.txt").write_text("not an image")
            crops = target.discover_crop_references(root, 1)
            self.assertEqual([p["image_path"].name for p in crops], ["a_crop_02_hanging_object.jpg"])
            self.assertEqual(target.discover_crop_references(root, 0), [])
            # 包括超出发送上限的小图，也不得混入负例。
            self.assertEqual(
                [p["image_path"].name for p in target.discover_negative_references(root, 6)],
                ["negative.jpg"],
            )

    def test_crop_content_and_selection_change_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = reference_config(root)
            ref = Path(config["reference_image_dir"])
            make_reference_pair(ref, "positive")
            crop = make_image(ref / "positive_crop_01_hanging_object.png", (40, 80))
            make_image(ref / "negative.jpg")
            target.prepare_reference_examples(config)
            self.assertEqual(len(config["reference_crops"]), 1)
            self.assertEqual(len(config["reference_negatives"]), 1)
            before = config["reference_manifest_sha256"]
            Image.new("RGB", (40, 80), "red").save(crop)
            target.prepare_reference_examples(config)
            changed = config["reference_manifest_sha256"]
            self.assertNotEqual(before, changed)
            config["max_reference_crops"] = 0
            target.prepare_reference_examples(config)
            self.assertNotEqual(changed, config["reference_manifest_sha256"])
            self.assertEqual(config["reference_crops"], [])
            self.assertEqual(len(config["reference_negatives"]), 1)

    def test_crop_only_and_negative_only_inputs_are_sent(self):
        for kind in ("reference_crops", "reference_negatives"):
            with self.subTest(kind=kind):
                config = {"labels": [{"name": LABEL}], kind: [{"rendered_data_url": "REF"}]}
                content = target.build_messages(config, "PROMPT", [], "FULL")
                self.assertEqual(
                    [b["image_url"]["url"] for b in content if b["type"] == "image_url"],
                    ["REF", "FULL"],
                )

    def test_zero_positive_limit_allows_crop_only_preparation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = reference_config(Path(temp_dir), max_reference_examples=0)
            ref = Path(config["reference_image_dir"])
            make_reference_pair(ref, "positive")
            make_image(ref / "positive_crop_01_hanging_object.png")
            target.prepare_reference_examples(config)
            self.assertEqual(config["reference_examples"], [])
            self.assertEqual(len(config["reference_crops"]), 1)
