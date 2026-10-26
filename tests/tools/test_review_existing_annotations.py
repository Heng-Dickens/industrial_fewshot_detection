import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import industrial_fewshot.legacy.review_annotations as target


def make_config(root: Path) -> dict:
    dataset = root / "dataset"
    dataset.mkdir()
    return {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "test-model",
        "dataset_dir": str(dataset),
        "output_dir": str(root / "corrected"),
        "audit_dir": str(root / "audit"),
        "labels": [
            {"name": "A", "description": "类别A"},
            {"name": "B", "description": "类别B"},
            {"name": "C", "description": "类别C"},
        ],
        "label_definitions_confirmed": True,
        "apply_remove": True,
        "apply_relabel": True,
        "apply_missing": False,
        "review_empty_annotations": False,
        "save_review_images": True,
        "stream": False,
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 0.9,
        "missing_max_iou": 0.25,
        "concurrency": 1,
        "max_attempts": 1,
        "max_tokens": 256,
        "structured_output_mode": "json_schema",
    }


def document() -> dict:
    return {
        "version": "5.1.1",
        "flags": {},
        "shapes": [
            {"label": "A", "points": [[10, 10], [20, 20]], "shape_type": "rectangle", "flags": {}},
            {"label": "B", "points": [[30, 30], [40, 40]], "shape_type": "rectangle", "flags": {}},
            {"label": "C", "points": [[50, 50], [60, 60]], "shape_type": "rectangle", "flags": {}},
        ],
        "imagePath": "sample.jpg",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }


class ReviewExistingAnnotationsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        raw = make_config(self.root)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        self.config = target.load_config(self.config_path)
        self.doc = document()
        self.candidates = target.extract_candidates(self.doc, {"A", "B", "C"})

    def tearDown(self):
        self.temp_dir.cleanup()

    def review_value(self):
        return {
            "existing_reviews": [
                {"object_id": "obj_0001", "decision": "remove", "corrected_label": "", "reason": "背景误检"},
                {"object_id": "obj_0002", "decision": "relabel", "corrected_label": "C", "reason": "类别应为C"},
                {"object_id": "obj_0003", "decision": "uncertain", "corrected_label": "", "reason": "局部遮挡"},
            ],
            "missing_objects": [
                {"label": "A", "bbox_2d": [800, 800, 900, 900], "reason": "明显漏标"}
            ],
            "image_reason": "完成已有框复核并检查漏标",
        }

    def test_prompt_prioritizes_false_positive_review_and_lists_every_id(self):
        prompt = target.build_prompt(self.config, self.candidates)
        self.assertIn("首要任务是删除", prompt)
        self.assertIn("第二任务才是补充", prompt)
        self.assertIn("obj_0001", prompt)
        self.assertIn("obj_0003", prompt)
        self.assertIn("uncertain 默认保留", prompt)

    def test_validate_review_requires_exact_existing_ids(self):
        value = self.review_value()
        value["existing_reviews"].pop()
        with self.assertRaisesRegex(ValueError, "未完整审核"):
            target.validate_review(value, self.candidates, {"A", "B", "C"})

    def test_validate_review_rejects_invalid_relabel(self):
        value = self.review_value()
        value["existing_reviews"][1]["corrected_label"] = "B"
        with self.assertRaisesRegex(ValueError, "corrected_label 无效"):
            target.validate_review(value, self.candidates, {"A", "B", "C"})

    def test_apply_review_removes_relabels_keeps_uncertain_and_gates_missing(self):
        review = target.validate_review(self.review_value(), self.candidates, {"A", "B", "C"})
        corrected, changes = target.apply_review(
            self.doc, self.candidates, review, 100, 100, self.config
        )
        self.assertEqual([shape["label"] for shape in corrected["shapes"]], ["C", "C"])
        self.assertEqual(self.doc["shapes"][1]["label"], "B")
        self.assertEqual(changes[2]["applied_action"], "kept_for_manual_review")
        self.assertEqual(changes[3]["applied_action"], "missing_suggested_not_applied")

        self.config["apply_missing"] = True
        corrected, changes = target.apply_review(
            self.doc, self.candidates, review, 100, 100, self.config
        )
        self.assertEqual(len(corrected["shapes"]), 3)
        self.assertEqual(changes[3]["applied_action"], "missing_added")

    def test_review_board_is_valid_jpeg(self):
        image = Image.new("RGB", (100, 100), "white")
        board = target.build_review_board(image, self.candidates)
        with Image.open(__import__("io").BytesIO(board)) as result:
            self.assertEqual(result.format, "JPEG")
            self.assertGreaterEqual(result.width, 100)

    def test_passthrough_does_not_overwrite_existing_output(self):
        source_json = Path(self.config["dataset_dir"]) / "empty.json"
        source_image = source_json.with_suffix(".jpg")
        Image.new("RGB", (20, 20), "white").save(source_image)
        empty = document()
        empty["shapes"] = []
        empty["imagePath"] = source_image.name
        source_json.write_text(json.dumps(empty), encoding="utf-8")
        destination = Path(self.config["output_dir"]) / "empty.json"
        destination.parent.mkdir(parents=True)
        destination.write_text('{"marker":"reviewed"}', encoding="utf-8")
        item = {
            "json_path": source_json,
            "image_path": source_image,
            "relative_json": Path("empty.json"),
            "candidate_count": 0,
        }
        copied = target.sync_passthrough_empty([item], self.config)
        self.assertEqual(copied, 0)
        self.assertEqual(json.loads(destination.read_text())["marker"], "reviewed")


class FakeCompletions:
    async def create(self, **request):
        self.request = request
        content = json.dumps({
            "existing_reviews": [
                {"object_id": "obj_0001", "decision": "keep", "corrected_label": "", "reason": "正确"}
            ],
            "missing_objects": [],
            "image_reason": "审核完成",
        }, ensure_ascii=False)
        message = SimpleNamespace(content=content, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class ReviewRequestTests(unittest.TestCase):
    def test_request_sends_original_and_review_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            raw = make_config(root)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            config = target.load_config(config_path)
            candidate = {"object_id": "obj_0001", "shape_index": 0, "label": "A", "bbox_pixels_xyxy": [1, 1, 10, 10]}
            client = FakeClient()
            result = asyncio.run(target.request_review(
                client, config, "prompt", "data:image/jpeg;base64,AA==",
                "data:image/jpeg;base64,AA==", [candidate]
            ))
            self.assertEqual(result["review"]["existing_reviews"][0]["decision"], "keep")
            self.assertEqual(len(client.completions.request["messages"][0]["content"]), 3)

    def test_request_sends_each_label_reference_image_before_review_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            raw = make_config(root)
            reference_dir = root / "references"
            reference_dir.mkdir()
            for label in ("A", "B", "C"):
                Image.new("RGB", (20, 20), "white").save(reference_dir / f"{label}.png")
            raw["reference_image_dir"] = str(reference_dir)
            raw["use_reference_images"] = True
            config_path = root / "config.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            config = target.load_config(config_path)
            references = target.prepare_reference_images(config)
            candidate = {
                "object_id": "obj_0001",
                "shape_index": 0,
                "label": "A",
                "bbox_pixels_xyxy": [1, 1, 10, 10],
            }
            client = FakeClient()
            asyncio.run(target.request_review(
                client, config, "prompt", "data:image/jpeg;base64,AA==",
                "data:image/jpeg;base64,AA==", [candidate], references
            ))
            content = client.completions.request["messages"][0]["content"]
            self.assertEqual(len(references), 3)
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 5)
            self.assertEqual(
                [item["text"] for item in content if item["type"] == "text"][:4],
                [
                    "下面先提供类别参考图。每张参考图前的文字标签与紧随其后的图片一一对应；请先阅读全部参考图，再处理后面的待审核图片。",
                    "类别参考图 1/3：A",
                    "类别参考图 2/3：B",
                    "类别参考图 3/3：C",
                ],
            )


if __name__ == "__main__":
    unittest.main()
