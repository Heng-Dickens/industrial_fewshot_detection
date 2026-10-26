import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageColor

import industrial_fewshot.prompts.references as references
import industrial_fewshot.data.prepare as target


def make_image(path, size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "#777777").save(path)
    return path


def shape(label="target", points=None, kind="rectangle"):
    return {"label": label, "shape_type": kind, "points": points or [[10, 20], [40, 60]]}


def annotate(image, shapes):
    path = image.with_suffix(".json")
    path.write_text(json.dumps({"shapes": shapes, "imageWidth": 100, "imageHeight": 80,
                               "imagePath": image.name, "imageData": "EMBEDDED_BYTES"}), encoding="utf-8")
    return path


class PrepareReferenceDataTest(unittest.TestCase):
    def test_selected_images_export_only_the_requested_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            annotate(make_image(root / "a/positive.jpg"), [shape()])
            make_image(root / "b/negative.jpg")
            make_image(root / "unused.jpg")
            report = target.prepare_dataset(root, selected_paths=[Path("a/positive.jpg"), Path("b/negative.jpg")])
            self.assertEqual(report["image_count"], 2)
            self.assertEqual(report["positive_image_count"], 1)
            self.assertEqual(report["negative_image_count"], 1)
            self.assertEqual({row["source_image"] for row in report["images"]}, {"a/positive.jpg", "b/negative.jpg"})
            self.assertFalse((Path(report["output_dir"]) / "negative/images/unused.jpg").exists())

    def test_selected_image_paths_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            make_image(root / "inside.jpg")
            make_image(Path(directory) / "outside.jpg")
            with self.assertRaisesRegex(ValueError, "输入目录"):
                target.prepare_dataset(root, selected_paths=[Path("../outside.jpg")])
            self.assertFalse(root.with_name("source_prepared").exists())

    def test_complete_split_exports_assets_preserves_sources_and_valid_json_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            positive = make_image(root / "positive.jpg")
            annotation = annotate(positive, [shape()])
            make_image(root / "missing.jpg")
            annotate(make_image(root / "empty.jpg"), [])
            before = {p: p.read_bytes() for p in root.iterdir()}
            report = target.prepare_dataset(root)
            output = Path(report["output_dir"])
            self.assertEqual(output, root.with_name("source_prepared").resolve())
            self.assertEqual((report["positive_image_count"], report["negative_image_count"], report["crop_count"]), (1, 2, 1))
            self.assertEqual((output / "positive/images/positive.jpg").read_bytes(), positive.read_bytes())
            self.assertEqual((output / "negative/images/missing.jpg").read_bytes(), before[root / "missing.jpg"])
            document = json.loads((output / "positive/annotations/positive.json").read_text(encoding="utf-8"))
            self.assertIsNone(document["imageData"])
            self.assertEqual((output / "positive/annotations" / document["imagePath"]).resolve(), (output / "positive/images/positive.jpg").resolve())
            self.assertTrue((output / "negative/annotations/empty.json").is_file())
            self.assertFalse((output / "negative/annotations/missing.json").exists())
            row = next(r for r in report["images"] if r["split"] == "positive")
            with Image.open(output / row["boxed_image"]) as boxed:
                self.assertEqual(boxed.size, (100, 80))
                self.assertEqual(boxed.getpixel((10, 50)), (255, 0, 0))
            with Image.open(output / row["crops"][0]["image"]) as crop:
                self.assertEqual(crop.size, (30, 40))
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertEqual(json.loads((output / "manifest.json").read_text(encoding="utf-8")), report)
            self.assertEqual(annotation.read_bytes(), before[annotation])

    def test_multiclass_colors_are_distinct_and_class_crops_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            annotate(make_image(root / "one.png"), [shape("a"), shape("b", [[55, 20], [90, 60]])])
            annotate(make_image(root / "two.png"), [shape("a")])
            report = target.prepare_dataset(root)
            classes = report["classes"]
            self.assertNotEqual(classes["a"]["color"], classes["b"]["color"])
            self.assertEqual(classes["a"]["object_count"], 2)
            output = Path(report["output_dir"])
            with Image.open(output / "positive/boxed_images/one_boxed.png") as boxed:
                self.assertEqual(boxed.getpixel((10, 50)), ImageColor.getrgb(classes["a"]["color"]))
                self.assertEqual(boxed.getpixel((55, 50)), ImageColor.getrgb(classes["b"]["color"]))
            with Image.open(output / "positive/boxed_images/two_boxed.png") as boxed:
                self.assertEqual(boxed.getpixel((10, 50)), ImageColor.getrgb(classes["a"]["color"]))
            self.assertEqual(len(list((output / "positive/crops/a").rglob("*.png"))), 2)
            self.assertEqual(len(list((output / "positive/crops/b").rglob("*.png"))), 1)

    def test_polygon_and_reversed_fractional_rectangle_use_covering_integer_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            annotate(make_image(root / "one.png"), [shape(points=[[40.4, 59.2], [10.2, 20.1]]),
                shape("polygon", [[55, 20], [90, 20], [70, 60]], "polygon")])
            report = target.prepare_dataset(root)
            row = report["images"][0]
            self.assertEqual(row["crops"][0]["crop_xyxy"], [10, 20, 41, 60])
            with Image.open(Path(report["output_dir"]) / row["crops"][0]["image"]) as crop:
                self.assertEqual(crop.size, (31, 40))
            self.assertEqual(row["crops"][1]["bbox_xyxy"], [55, 20, 90, 60])

    def test_known_derivatives_are_excluded_unless_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            annotate(make_image(root / "original.jpg"), [shape()])
            make_image(root / "original_crop_01_target.png")
            make_image(root / "original_boxed.png")
            report = target.prepare_dataset(root, dry_run=True)
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(report["negative_image_count"], 0)
            self.assertEqual(len(report["skipped_derived_images"]), 2)
            included = target.prepare_dataset(root, dry_run=True, include_derived=True)
            self.assertEqual(included["negative_image_count"], 2)
            self.assertFalse(Path(report["output_dir"]).exists())

    def test_invalid_annotation_fails_before_export_and_never_becomes_negative(self):
        cases = ["broken_json", {"shapes": "not a list"}, {"imageWidth": 10},
                 {"shapes": [shape(points=[[-1, 20], [40, 60]])]},
                 {"shapes": [shape(points=[[10, 20], [10, 60]])]},
                 {"shapes": [shape(kind="circle")]}, {"shapes": [shape(label="")]},
                 {"other": float("nan")}]
        for change in cases:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "source"
                make_image(root / "a_valid_negative.jpg")
                path = annotate(make_image(root / "z_bad.jpg"), [shape()])
                if change == "broken_json":
                    path.write_text("{broken", encoding="utf-8")
                else:
                    document = json.loads(path.read_text(encoding="utf-8"))
                    document.update(change)
                    path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    target.prepare_dataset(root)
                self.assertFalse(root.with_name("source_prepared").exists())

    def test_recursive_same_filenames_are_flat_with_safe_names_and_valid_json_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for folder in ("a", "b"):
                annotate(make_image(root / folder / "frame.jpg"), [shape()])
            report = target.prepare_dataset(root, recursive=True)
            output = Path(report["output_dir"])
            images = list((output / "positive/images").iterdir())
            self.assertEqual(len(images), 2)
            self.assertTrue(all(path.is_file() for path in images))
            self.assertEqual(len({path.stem for path in images}), 2)
            self.assertTrue(all(path.stem.startswith("frame__") for path in images))
            for row in report["images"]:
                annotation = output / row["annotation"]
                document = json.loads(annotation.read_text(encoding="utf-8"))
                self.assertEqual((annotation.parent / document["imagePath"]).resolve(), (output / row["image"]).resolve())
            self.assertEqual(len(list((output / "positive/crops").rglob("*.png"))), 2)
            make_image(root / "a/frame.png")
            with self.assertRaisesRegex(ValueError, "相同标注"):
                target.prepare_dataset(root, Path(directory) / "conflict", recursive=True)

    def test_recursive_unique_filenames_are_flat_in_every_asset_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            annotate(make_image(root / "00001/positive.jpg"), [shape()])
            make_image(root / "00002/negative.jpg")
            report = target.prepare_dataset(root, recursive=True)
            output = Path(report["output_dir"])
            self.assertTrue((output / "positive/images/positive.jpg").is_file())
            self.assertTrue((output / "negative/images/negative.jpg").is_file())
            self.assertTrue((output / "positive/annotations/positive.json").is_file())
            self.assertTrue((output / "positive/boxed_images/positive_boxed.png").is_file())
            self.assertTrue((output / "positive/crops/target/positive_crop_001_target.png").is_file())
            self.assertEqual({row["source_image"] for row in report["images"]}, {"00001/positive.jpg", "00002/negative.jpg"})
            for folder in (output / "positive/images", output / "negative/images", output / "positive/annotations",
                           output / "positive/boxed_images", output / "positive/crops/target"):
                self.assertTrue(all(path.is_file() for path in folder.iterdir()))

    def test_windows_safe_class_names_and_many_colors_have_no_collisions(self):
        labels = {"a/b", "a:b", "CON", "NUL.txt", "Part", "part", "中文 类别", *[f"class{i}" for i in range(25)]}
        classes = target.class_metadata(labels)
        self.assertEqual(len({info["crop_directory"].casefold() for info in classes.values()}), len(labels))
        self.assertEqual(len({info["color"] for info in classes.values()}), len(labels))
        self.assertEqual(classes, target.class_metadata(set(reversed(sorted(labels)))))
        self.assertTrue(classes["CON"]["crop_directory"].startswith("class_"))
        self.assertTrue(classes["NUL.txt"]["crop_directory"].startswith("class_"))
        self.assertNotIn("/", classes["a/b"]["crop_directory"])

    def test_output_inside_source_and_nonempty_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            make_image(root / "negative.jpg")
            for output in (root, root / "inside"):
                with self.assertRaisesRegex(ValueError, "输入目录之外"):
                    target.prepare_dataset(root, output)
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "非空"):
                target.prepare_dataset(root, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_transparency_and_source_resolution_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            path = root / "transparent.png"
            Image.new("RGBA", (100, 80), (10, 20, 30, 0)).save(path)
            annotate(path, [shape()])
            report = target.prepare_dataset(root)
            output = Path(report["output_dir"])
            with Image.open(output / report["images"][0]["boxed_image"]) as boxed:
                self.assertEqual(boxed.size, (100, 80))
                self.assertEqual(boxed.getpixel((99, 79))[3], 0)
            with Image.open(output / report["images"][0]["crops"][0]["image"]) as crop:
                self.assertEqual(crop.mode, "RGBA")

    def test_prepared_references_do_not_misclassify_positive_assets_as_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            annotate(make_image(root / "positive.jpg"), [shape()])
            make_image(root / "missing.jpg")
            annotate(make_image(root / "empty.jpg"), [])
            output = Path(target.prepare_dataset(root)["output_dir"])
            self.assertEqual(len(references.discover_reference_pairs(output, 10)), 1)
            self.assertEqual(len(references.discover_json_references(output, 10)), 1)
            self.assertEqual(len(references.discover_crop_references(output, 10)), 1)
            self.assertEqual(len(references.discover_negative_references(output, 10)), 2)

    def test_cli_dry_run_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            make_image(root / "negative.jpg")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = target.main(["--input-dir", str(root), "--dry-run"])
            report = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(report["negative_image_count"], 1)
            self.assertFalse(Path(report["output_dir"]).exists())
