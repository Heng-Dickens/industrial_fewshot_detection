import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import industrial_fewshot.data.clean_annotations as target


def make_image(path: Path, size: tuple[int, int] = (640, 360)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (180, 180, 60)).save(path)
    return path


def make_shapes(*, empty: bool) -> list[dict]:
    if empty:
        return []
    return [
        {
            "label": "DiaoGou",
            "points": [[10.0, 20.0], [300.0, 320.0]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
    ]


def write_annotation(
    path: Path, *, empty: bool, with_shapes_key: bool = True
) -> Path:
    document: dict = {
        "version": "6.3.1",
        "flags": {},
        "imageData": None,
        "imagePath": path.with_suffix(".jpg").name,
        "imageHeight": 360,
        "imageWidth": 640,
    }
    if with_shapes_key:
        document["shapes"] = make_shapes(empty=empty)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def make_args(**overrides) -> SimpleNamespace:
    values = {
        "config": None,
        "root": None,
        "extensions": None,
        "dry_run": False,
        "list_limit": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def load_extensions() -> set[str]:
    return {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InspectAnnotationTest(unittest.TestCase):
    def test_empty_shapes_is_marked_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "frame.jpg")
            json_path = write_annotation(root / "frame.json", empty=True)
            verdict, _ = target.inspect_annotation(json_path, root, load_extensions())
            self.assertEqual(verdict, "empty")

    def test_shapes_with_box_is_kept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "frame.jpg")
            json_path = write_annotation(root / "frame.json", empty=False)
            verdict, _ = target.inspect_annotation(json_path, root, load_extensions())
            self.assertEqual(verdict, "kept")

    def test_json_without_matching_image_is_protected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            # 模拟汇总文件：shapes 为空但没有同名图片
            write_annotation(root / "annotations.json", empty=True)
            verdict, detail = target.inspect_annotation(
                root / "annotations.json", root, load_extensions()
            )
            self.assertEqual(verdict, "missing_image")
            self.assertIn("annotations.json", detail)

    def test_json_without_shapes_key_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "frame.jpg")
            json_path = write_annotation(
                root / "frame.json", empty=True, with_shapes_key=False
            )
            verdict, _ = target.inspect_annotation(json_path, root, load_extensions())
            self.assertEqual(verdict, "invalid")

    def test_broken_json_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "frame.jpg")
            json_path = root / "frame.json"
            json_path.write_text("{not json", encoding="utf-8")
            verdict, _ = target.inspect_annotation(json_path, root, load_extensions())
            self.assertEqual(verdict, "invalid")

    def test_image_extension_case_and_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "frame.png")
            json_path = write_annotation(root / "frame.json", empty=True)
            verdict, _ = target.inspect_annotation(json_path, root, load_extensions())
            self.assertEqual(verdict, "empty")


class CleanDirectoryTest(unittest.TestCase):
    def test_dry_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "a.jpg")
            write_annotation(root / "a.json", empty=True)
            result = target.clean_directory(root, load_extensions(), True)
            self.assertEqual(result["counts"]["empty_deleted"], 1)
            self.assertTrue((root / "a.json").is_file())

    def test_apply_deletes_only_empty_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "empty.jpg")
            empty_json = write_annotation(root / "empty.json", empty=True)
            make_image(root / "target.jpg")
            kept_json = write_annotation(root / "target.json", empty=False)
            # 汇总文件没有同名图片，必须保留
            summary_json = write_annotation(root / "annotations.json", empty=True)

            result = target.clean_directory(root, load_extensions(), False)
            self.assertEqual(result["counts"]["empty_deleted"], 1)
            self.assertFalse(empty_json.is_file())
            self.assertTrue(kept_json.is_file())
            self.assertTrue(summary_json.is_file())
            self.assertEqual(result["counts"]["missing_image"], 1)
            self.assertEqual(result["counts"]["kept"], 1)

    def test_recurses_into_subdirectories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "00000" / "a.jpg")
            nested = write_annotation(root / "00000" / "a.json", empty=True)
            result = target.clean_directory(root, load_extensions(), False)
            self.assertEqual(result["counts"]["empty_deleted"], 1)
            self.assertFalse(nested.is_file())

    def test_counts_are_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "a.jpg")
            write_annotation(root / "a.json", empty=True)
            make_image(root / "b.jpg")
            write_annotation(root / "b.json", empty=False)
            make_image(root / "c.jpg")
            (root / "c.json").write_text("{broken", encoding="utf-8")
            write_annotation(root / "d.json", empty=True)  # 无同名图片

            result = target.clean_directory(root, load_extensions(), True)
            counts = result["counts"]
            self.assertEqual(counts["json_scanned"], 4)
            self.assertEqual(counts["empty_deleted"], 1)
            self.assertEqual(counts["kept"], 1)
            self.assertEqual(counts["invalid"], 1)
            self.assertEqual(counts["missing_image"], 1)
            self.assertEqual(counts["failed"], 0)


class LoadTargetDirTest(unittest.TestCase):
    def test_root_argument_uses_default_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            directory = root / "images"
            directory.mkdir()
            args = make_args(root=directory)
            image_dir, extensions = target.load_target_dir(args)
            self.assertEqual(image_dir, directory.resolve())
            self.assertIn(".jpg", extensions)

    def test_extensions_argument_is_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            args = make_args(root=root, extensions=["JPG", ".png"])
            _, extensions = target.load_target_dir(args)
            self.assertEqual(extensions, {".jpg", ".png"})

    def test_config_image_dir_used_without_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            image_dir = root / "images"
            image_dir.mkdir()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "image_dir": str(image_dir),
                        "extensions": [".jpg", ".png"],
                    }
                ),
                encoding="utf-8",
            )
            resolved, extensions = target.load_target_dir(
                make_args(config=config_path)
            )
            self.assertEqual(resolved, image_dir.resolve())
            self.assertEqual(extensions, {".jpg", ".png"})

    def test_missing_image_dir_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"labels": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image_dir"):
                target.load_target_dir(make_args(config=config_path))

    def test_non_existent_root_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            with self.assertRaisesRegex(ValueError, "目录不存在"):
                target.load_target_dir(make_args(root=root / "not_there"))

    def test_extensions_default_when_config_omits_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            image_dir = root / "images"
            image_dir.mkdir()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"image_dir": str(image_dir)}), encoding="utf-8"
            )
            _, extensions = target.load_target_dir(make_args(config=config_path))
            self.assertIn(".jpeg", extensions)
            self.assertIn(".webp", extensions)


class RunTest(unittest.TestCase):
    def test_run_returns_zero_and_respects_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            make_image(root / "frame.jpg")
            json_path = write_annotation(root / "frame.json", empty=True)
            exit_code = target.run(make_args(root=root, dry_run=True, list_limit=0))
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_path.is_file())

            exit_code = target.run(make_args(root=root, dry_run=False, list_limit=0))
            self.assertEqual(exit_code, 0)
            self.assertFalse(json_path.is_file())

    def test_negative_list_limit_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            with self.assertRaisesRegex(ValueError, "list-limit"):
                target.run(make_args(root=root, list_limit=-1))


if __name__ == "__main__":
    unittest.main()
