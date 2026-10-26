import argparse
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

import industrial_fewshot.data.exports as target


class AnnotationToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.image_root = self.root / "source_images"
        self.image_path = self.image_root / "nested" / "sample.jpg"
        self.image_path.parent.mkdir(parents=True)
        Image.new("RGB", (200, 100), "white").save(self.image_path)
        self.empty_image_path = self.image_root / "z_empty.jpg"
        Image.new("RGB", (200, 100), "white").save(self.empty_image_path)
        self.input_path = self.root / "annotations.json"
        self.document = {
            "dataset": {"image_root": str(self.image_root)},
            "inference": {
                "labels": [
                    {"name": "green_tractor", "description": "绿色牵引车"}
                ]
            },
            "annotations": [
                {
                    "relative_path": "nested/sample.jpg",
                    "image_path": str(self.image_path),
                    "width": 200,
                    "height": 100,
                    "status": "success",
                    "model": "test-model",
                    "prompt_version": "test-v2",
                    "annotation_status": "annotated",
                    "target_count": 1,
                    "reason_code": "target_detected",
                    "reason": "检测到清晰可见的绿色牵引车",
                    "raw_response": (
                        '{"annotation_status":"annotated",'
                        '"target_count":1,'
                        '"reason_code":"target_detected",'
                        '"reason":"检测到清晰可见的绿色牵引车",'
                        '"objects":[{"label":"green_tractor",'
                        '"bbox_2d":[100,200,800,900]}]}'
                    ),
                    "initial_raw_response": "首轮原始回复：一个目标",
                    "review_performed": True,
                    "objects": [
                        {
                            "label": "green_tractor",
                            "bbox_2d": [100, 200, 800, 900],
                        }
                    ],
                },
                {
                    "relative_path": "z_empty.jpg",
                    "image_path": str(self.empty_image_path),
                    "width": 200,
                    "height": 100,
                    "status": "success",
                    "model": "test-model",
                    "prompt_version": "test-v2",
                    "annotation_status": "no_target",
                    "target_count": 0,
                    "reason_code": "no_target",
                    "reason": "图像清晰，但没有指定类型的车辆",
                    "raw_response": (
                        '{"annotation_status":"no_target",'
                        '"target_count":0,'
                        '"reason_code":"no_target",'
                        '"reason":"图像清晰，但没有指定类型的车辆",'
                        '"objects":[]}'
                    ),
                    "objects": [],
                },
                {
                    "relative_path": "failed.jpg",
                    "width": 200,
                    "height": 100,
                    "status": "failed",
                    "objects": [],
                },
            ],
        }
        self.input_path.write_text(
            json.dumps(self.document, ensure_ascii=False), encoding="utf-8"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_args(self, output_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            input=self.input_path,
            config=self.root / "unused.json",
            output_dir=output_dir,
            formats=["preview", "yolo", "coco"],
            image_mode="copy",
            font=None,
            font_size=18,
            line_width=3,
            limit=None,
        )

    def test_all_formats_export_expected_coordinates_and_files(self):
        output_dir = self.root / "export"
        result = target.run(self.make_args(output_dir))

        self.assertEqual(result, output_dir)
        preview = output_dir / "preview" / "nested" / "sample.jpg"
        self.assertTrue(preview.is_file())
        with Image.open(preview) as image:
            self.assertNotEqual(image.getpixel((20, 20)), (255, 255, 255))

        yolo_label = output_dir / "yolo" / "labels" / "nested" / "sample.txt"
        self.assertEqual(
            yolo_label.read_text(encoding="utf-8").strip(),
            "0 0.450000 0.550000 0.700000 0.700000",
        )
        self.assertTrue(
            (output_dir / "yolo" / "images" / "nested" / "sample.jpg").is_file()
        )

        coco_path = output_dir / "coco" / "annotations" / "instances.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        self.assertEqual(coco["images"][0]["file_name"], "images/nested/sample.jpg")
        self.assertEqual(coco["annotations"][0]["bbox"], [20.0, 20.0, 140.0, 70.0])
        self.assertEqual(coco["annotations"][0]["area"], 9800.0)
        self.assertEqual(coco["categories"][0]["name"], "green_tractor")

        report = json.loads(
            (output_dir / "conversion_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["failed_records_skipped"], 1)
        self.assertEqual(report["results"]["preview"]["objects"], 1)
        self.assertEqual(report["results"]["yolo"]["objects"], 1)
        self.assertEqual(report["results"]["coco"]["objects"], 1)

    def test_nonempty_output_directory_is_rejected(self):
        output_dir = self.root / "existing"
        output_dir.mkdir()
        (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "输出目录非空"):
            target.run(self.make_args(output_dir))

        self.assertEqual(
            (output_dir / "keep.txt").read_text(encoding="utf-8"), "user data"
        )

    def test_default_input_uses_image_directory_identifier_suffix(self):
        image_dir = self.root / "00042"
        config_path = self.root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "image_dir": str(image_dir),
                    "output_json": str(self.root / "annotations" / "annotations.json"),
                    "append_image_dir_suffix": True,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(input=None, config=config_path)

        resolved = target.resolve_input_path(args)

        self.assertEqual(resolved.name, "annotations_00042.json")

    def test_labelme_voc_yolo_and_model_replies_match(self):
        output_dir = self.root / "split_export"
        args = self.make_args(output_dir)
        args.formats = ["labelme", "voc", "yolo", "replies"]

        target.run(args)

        labelme_path = output_dir / "labelme" / "nested" / "sample.json"
        labelme = json.loads(labelme_path.read_text(encoding="utf-8"))
        self.assertEqual(labelme["version"], "5.1.1")
        self.assertEqual(labelme["imagePath"], "sample.jpg")
        self.assertIsNone(labelme["imageData"])
        self.assertEqual(labelme["imageWidth"], 200)
        self.assertEqual(labelme["imageHeight"], 100)
        self.assertEqual(labelme["shapes"][0]["label"], "green_tractor")
        self.assertEqual(
            labelme["shapes"][0]["points"], [[20.0, 20.0], [160.0, 90.0]]
        )
        self.assertTrue(
            (output_dir / "labelme" / "nested" / "sample.jpg").is_file()
        )
        empty_labelme = json.loads(
            (output_dir / "labelme" / "z_empty.json").read_text(encoding="utf-8")
        )
        self.assertEqual(empty_labelme["shapes"], [])

        voc_path = output_dir / "voc" / "Annotations" / "nested" / "sample.xml"
        voc = ET.parse(voc_path).getroot()
        self.assertEqual(voc.findtext("folder"), "nested")
        self.assertEqual(voc.findtext("filename"), "sample.jpg")
        self.assertEqual(voc.findtext("size/width"), "200")
        self.assertEqual(voc.findtext("size/height"), "100")
        self.assertEqual(voc.findtext("size/depth"), "3")
        self.assertEqual(voc.findtext("object/name"), "green_tractor")
        self.assertEqual(voc.findtext("object/bndbox/xmin"), "20")
        self.assertEqual(voc.findtext("object/bndbox/ymin"), "20")
        self.assertEqual(voc.findtext("object/bndbox/xmax"), "160")
        self.assertEqual(voc.findtext("object/bndbox/ymax"), "90")
        self.assertTrue(
            (
                output_dir
                / "voc"
                / "JPEGImages"
                / "nested"
                / "sample.jpg"
            ).is_file()
        )
        empty_voc = ET.parse(
            output_dir / "voc" / "Annotations" / "z_empty.xml"
        ).getroot()
        self.assertIsNone(empty_voc.find("object"))

        yolo = (
            output_dir / "yolo" / "labels" / "nested" / "sample.txt"
        ).read_text(encoding="utf-8").strip()
        self.assertEqual(yolo, "0 0.450000 0.550000 0.700000 0.700000")
        self.assertEqual(
            (output_dir / "yolo" / "labels" / "z_empty.txt").read_text(
                encoding="utf-8"
            ),
            "",
        )

        first_reply_path = output_dir / "model_replies" / "nested" / "sample.json"
        second_reply_path = output_dir / "model_replies" / "z_empty.json"
        self.assertTrue(first_reply_path.is_file())
        self.assertTrue(second_reply_path.is_file())
        first_reply = json.loads(first_reply_path.read_text(encoding="utf-8"))
        second_reply = json.loads(second_reply_path.read_text(encoding="utf-8"))
        self.assertEqual(first_reply["relative_path"], "nested/sample.jpg")
        self.assertEqual(first_reply["model"], "test-model")
        self.assertEqual(first_reply["target_count"], 1)
        self.assertEqual(first_reply["reason_code"], "target_detected")
        self.assertEqual(
            first_reply["raw_response"],
            self.document["annotations"][0]["raw_response"],
        )
        self.assertEqual(first_reply["initial_raw_response"], "首轮原始回复：一个目标")
        self.assertEqual(second_reply["reason_code"], "no_target")
        self.assertEqual(
            second_reply["raw_response"],
            self.document["annotations"][1]["raw_response"],
        )

        report = json.loads(
            (output_dir / "conversion_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["results"]["labelme"]["objects"], 1)
        self.assertEqual(report["results"]["voc"]["objects"], 1)
        self.assertEqual(report["results"]["yolo"]["objects"], 1)
        self.assertEqual(report["results"]["replies"]["files"], 2)
        self.assertEqual(report["results"]["replies"]["initial_raw_responses"], 1)
        self.assertEqual(report["results"]["replies"]["final_raw_responses"], 2)


if __name__ == "__main__":
    unittest.main()
