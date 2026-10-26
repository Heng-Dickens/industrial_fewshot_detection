import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import industrial_fewshot.legacy.batch_annotate as target


def make_config(root: Path) -> dict:
    return {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "test-model",
        "image_dir": str(root),
        "output_json": str(root / "annotations.json"),
        "checkpoint_jsonl": str(root / "checkpoint.jsonl"),
        "concurrency": 2,
        "max_attempts": 1,
        "retry_delay_seconds": 0,
        "max_tokens": 128,
        "temperature": 0.25,
        "top_p": 0.85,
        "enable_thinking": False,
        "stream": False,
        "enable_completeness_review": False,
        "prompt_version": "test-v1",
        "structured_output_mode": "json_schema",
        "labels": [
            {"name": "green_tractor", "description": "绿色牵引车"}
        ],
    }


class FakeCompletions:
    async def create(self, **request):
        self.request = request
        message = SimpleNamespace(
            content=(
                '{"annotation_status":"annotated",'
                '"reason_code":"target_detected",'
                '"reason":"检测到清晰可见的绿色牵引车",'
                '"target_count":1,'
                '"objects":[{"label":"green_tractor",'
                '"bbox_2d":[100,200,800,900]}]}'
            )
        )
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None)


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class SequenceCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        content = self.contents[len(self.requests) - 1]
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None)


class SequenceClient:
    def __init__(self, contents):
        self.completions = SequenceCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


class FakeAsyncStream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeStreamingCompletions:
    async def create(self, **request):
        self.request = request
        content = (
            '{"annotation_status":"annotated","reason_code":"target_detected",'
            '"reason":"流式检测到目标","target_count":1,'
            '"objects":[{"label":"green_tractor",'
            '"bbox_2d":[100,200,800,900]}]}'
        )
        split_at = len(content) // 2
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None, reasoning_content="正在检查全图"
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=content[:split_at], reasoning_content=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=content[split_at:], reasoning_content=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(choices=[], usage=FakeUsage()),
        ]
        return FakeAsyncStream(chunks)


class FakeStreamingClient:
    def __init__(self):
        self.completions = FakeStreamingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class BatchAnnotateTests(unittest.TestCase):
    def test_load_config_reads_generation_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            raw_config = make_config(root)
            raw_config.update(
                {
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "enable_thinking": True,
                    "stream": True,
                }
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(raw_config, ensure_ascii=False), encoding="utf-8"
            )
            args = SimpleNamespace(
                config=config_path,
                image_dir=None,
                output=None,
                checkpoint=None,
                concurrency=None,
            )

            config = target.load_config(args)

            self.assertEqual(config["temperature"], 0.6)
            self.assertEqual(config["top_p"], 0.9)
            self.assertTrue(config["enable_thinking"])
            self.assertTrue(config["stream"])

    def test_load_config_rejects_string_boolean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            raw_config = make_config(root)
            raw_config["enable_thinking"] = "false"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(raw_config), encoding="utf-8")
            args = SimpleNamespace(
                config=config_path,
                image_dir=None,
                output=None,
                checkpoint=None,
                concurrency=None,
            )

            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                target.load_config(args)

    def test_load_config_appends_image_directory_identifier_to_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            image_dir = root / "00042"
            image_dir.mkdir()
            raw_config = make_config(root)
            raw_config.update(
                {
                    "image_dir": str(image_dir),
                    "output_json": str(root / "out" / "annotations.json"),
                    "checkpoint_jsonl": str(
                        root / "out" / "annotations.checkpoint.jsonl"
                    ),
                    "append_image_dir_suffix": True,
                }
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(raw_config, ensure_ascii=False), encoding="utf-8"
            )
            args = SimpleNamespace(
                config=config_path,
                image_dir=None,
                output=None,
                checkpoint=None,
                concurrency=None,
            )

            config = target.load_config(args)

            self.assertEqual(config["dataset_id"], "00042")
            self.assertEqual(
                Path(config["output_json"]).name, "annotations_00042.json"
            )
            self.assertEqual(
                Path(config["checkpoint_jsonl"]).name,
                "annotations_00042.checkpoint.jsonl",
            )
            self.assertEqual(
                target.append_dataset_suffix(
                    Path(config["output_json"]), "00042"
                ),
                Path(config["output_json"]),
            )

    def test_clean_json_response_accepts_markdown_fence(self):
        result = target.clean_json_response(
            '```json\n{"objects": []}\n```'
        )
        self.assertEqual(result, {"objects": []})

    def test_validate_prediction_converts_to_pixel_coordinates(self):
        result = target.validate_prediction(
            {
                "annotation_status": "annotated",
                "reason_code": "target_detected",
                "reason": "检测到目标",
                "target_count": 1,
                "objects": [
                    {
                        "label": "green_tractor",
                        "bbox_2d": [100, 200, 800, 900],
                    }
                ]
            },
            {"green_tractor"},
            1920,
            1080,
        )
        self.assertEqual(
            result["objects"][0]["bbox_pixels_xyxy"], [192, 216, 1536, 972]
        )

    def test_validate_prediction_rejects_invalid_box(self):
        with self.assertRaisesRegex(ValueError, "坐标顺序无效"):
            target.validate_prediction(
                {
                    "annotation_status": "annotated",
                    "reason_code": "target_detected",
                    "reason": "检测到目标",
                    "target_count": 1,
                    "objects": [
                        {
                            "label": "green_tractor",
                            "bbox_2d": [800, 200, 100, 900],
                        }
                    ]
                },
                {"green_tractor"},
                1920,
                1080,
            )

    def test_validate_prediction_accepts_no_target_reason(self):
        result = target.validate_prediction(
            {
                "annotation_status": "no_target",
                "reason_code": "no_target",
                "reason": "图像清晰，但没有检测到指定类型的车辆",
                "target_count": 0,
                "objects": [],
            },
            {"green_tractor"},
            1920,
            1080,
        )
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["annotation_status"], "no_target")
        self.assertEqual(result["reason_code"], "no_target")

    def test_validate_prediction_accepts_unable_reason(self):
        result = target.validate_prediction(
            {
                "annotation_status": "unable_to_determine",
                "reason_code": "occluded",
                "reason": "疑似车辆超过三分之一被设备遮挡，无法可靠判断",
                "target_count": 0,
                "objects": [],
            },
            {"green_tractor"},
            1920,
            1080,
        )
        self.assertEqual(result["reason_code"], "occluded")

    def test_validate_prediction_rejects_empty_annotated_result(self):
        with self.assertRaisesRegex(ValueError, "objects 为空"):
            target.validate_prediction(
                {
                    "annotation_status": "annotated",
                    "reason_code": "target_detected",
                    "reason": "检测到目标",
                    "target_count": 0,
                    "objects": [],
                },
                {"green_tractor"},
                1920,
                1080,
            )

    def test_validate_prediction_rejects_target_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "与 objects 数量 1 不一致"):
            target.validate_prediction(
                {
                    "annotation_status": "annotated",
                    "reason_code": "target_detected",
                    "reason": "检测到目标",
                    "target_count": 2,
                    "objects": [
                        {
                            "label": "green_tractor",
                            "bbox_2d": [100, 200, 800, 900],
                        }
                    ],
                },
                {"green_tractor"},
                1920,
                1080,
            )

    def test_annotate_one_records_image_address_and_annotation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "nested" / "sample.jpg"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            config = make_config(root)
            client = FakeClient()

            record = asyncio.run(
                target.annotate_one(
                    client,
                    image_path.resolve(),
                    root.resolve(),
                    config,
                    target.build_prompt(config),
                    target.build_response_format(config),
                )
            )

            self.assertEqual(record["status"], "success")
            self.assertEqual(record["relative_path"], "nested/sample.jpg")
            self.assertEqual(record["image_path"], str(image_path.resolve()))
            self.assertEqual(record["width"], 200)
            self.assertEqual(record["height"], 100)
            self.assertEqual(record["annotation_status"], "annotated")
            self.assertEqual(record["target_count"], 1)
            self.assertEqual(record["reason_code"], "target_detected")
            self.assertEqual(record["reason"], "检测到清晰可见的绿色牵引车")
            self.assertEqual(
                record["objects"][0]["bbox_pixels_xyxy"], [20, 20, 160, 90]
            )
            request = client.completions.request
            self.assertEqual(request["temperature"], 0.25)
            self.assertEqual(request["top_p"], 0.85)
            self.assertFalse(request["stream"])
            self.assertFalse(
                request["extra_body"]["chat_template_kwargs"]["enable_thinking"]
            )

    def test_streaming_response_is_assembled_before_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            image_path = root / "stream.jpg"
            Image.new("RGB", (200, 100), "white").save(image_path)
            config = make_config(root)
            config["stream"] = True
            config["enable_thinking"] = True
            client = FakeStreamingClient()

            record = asyncio.run(
                target.annotate_one(
                    client,
                    image_path,
                    root,
                    config,
                    target.build_prompt(config),
                    target.build_response_format(config),
                )
            )

            self.assertEqual(record["status"], "success")
            self.assertEqual(record["target_count"], 1)
            self.assertEqual(record["reasoning_content"], "正在检查全图")
            self.assertEqual(record["finish_reason"], "stop")
            self.assertEqual(record["usage"]["total_tokens"], 30)
            request = client.completions.request
            self.assertTrue(request["stream"])
            self.assertEqual(request["stream_options"], {"include_usage": True})
            self.assertTrue(
                request["extra_body"]["chat_template_kwargs"]["enable_thinking"]
            )

    def test_completeness_review_replaces_initial_result_with_all_instances(self):
        first = json.dumps(
            {
                "annotation_status": "annotated",
                "target_count": 1,
                "reason_code": "target_detected",
                "reason": "第一遍检测到一个目标",
                "objects": [
                    {"label": "green_tractor", "bbox_2d": [50, 100, 400, 900]}
                ],
            },
            ensure_ascii=False,
        )
        reviewed = json.dumps(
            {
                "annotation_status": "annotated",
                "target_count": 2,
                "reason_code": "target_detected",
                "reason": "完整复核后检测到两个独立目标",
                "objects": [
                    {"label": "green_tractor", "bbox_2d": [50, 100, 400, 900]},
                    {"label": "green_tractor", "bbox_2d": [550, 120, 900, 880]},
                ],
            },
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            image_path = root / "two.jpg"
            Image.new("RGB", (200, 100), "white").save(image_path)
            config = make_config(root)
            config["enable_completeness_review"] = True
            client = SequenceClient([first, reviewed])

            record = asyncio.run(
                target.annotate_one(
                    client,
                    image_path,
                    root,
                    config,
                    target.build_prompt(config),
                    target.build_response_format(config),
                )
            )

            self.assertEqual(len(client.completions.requests), 2)
            self.assertEqual(record["target_count"], 2)
            self.assertEqual(len(record["objects"]), 2)
            self.assertEqual(record["initial_annotation"]["target_count"], 1)
            self.assertEqual(record["initial_raw_response"], first)
            self.assertEqual(record["raw_response"], reviewed)
            self.assertTrue(record["review_attempted"])
            self.assertTrue(record["review_performed"])
            self.assertTrue(record["review_changed_target_count"])
            review_text = client.completions.requests[1]["messages"][0]["content"][0]["text"]
            self.assertIn("完整最终列表", review_text)
            self.assertIn("不能只输出相对第一遍新增的目标", review_text)

    def test_summary_records_image_and_annotation_addresses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            config = make_config(root)
            checkpoint = Path(config["checkpoint_jsonl"])
            record = {
                "relative_path": "sample.jpg",
                "image_path": str(root / "sample.jpg"),
                "status": "success",
                "model": config["model"],
                "prompt_version": config["prompt_version"],
                "annotation_status": "no_target",
                "reason_code": "no_target",
                "reason": "图像清晰，但没有检测到指定目标",
                "target_count": 0,
                "objects": [],
                "generation_settings": target.generation_settings(config),
            }
            checkpoint.write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            summary = target.build_summary(config, image_count=1, scheduled_count=1)

            self.assertEqual(summary["dataset"]["image_root"], str(root))
            self.assertEqual(
                summary["dataset"]["annotation_file"],
                str(root / "annotations.json"),
            )
            self.assertEqual(
                summary["annotations"][0]["image_path"], str(root / "sample.jpg")
            )
            self.assertEqual(
                summary["statistics"]["no_annotation_reason_counts"],
                {"no_target": 1},
            )
            self.assertEqual(summary["statistics"]["current_prompt_version_count"], 1)

    def test_prompt_version_change_invalidates_checkpoint_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(Path(temp_dir))
            record = {
                "status": "success",
                "model": config["model"],
                "prompt_version": config["prompt_version"],
                "annotation_status": "no_target",
                "reason_code": "no_target",
                "reason": "没有检测到目标",
                "target_count": 0,
                "objects": [],
                "generation_settings": target.generation_settings(config),
            }
            self.assertTrue(target.is_reusable_success(record, config))

            config["enable_completeness_review"] = True
            self.assertFalse(target.is_reusable_success(record, config))
            record["review_performed"] = True
            self.assertTrue(target.is_reusable_success(record, config))
            config["enable_completeness_review"] = False

            record["prompt_version"] = "old-prompt"
            self.assertFalse(target.is_reusable_success(record, config))

            record["prompt_version"] = config["prompt_version"]
            record.pop("reason")
            self.assertFalse(target.is_reusable_success(record, config))


if __name__ == "__main__":
    unittest.main()
