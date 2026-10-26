#!/usr/bin/env python3
"""Organize LabelMe references into positive/negative samples and input assets."""
from __future__ import annotations

import argparse
import colorsys
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from PIL import Image, ImageDraw

try:
    from industrial_fewshot.data.exports import load_font
except ImportError:  # python tool/prepare_reference_data.py
    from industrial_fewshot.data.exports import load_font

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCHEMA = "annotationtool.reference_dataset.v1"
LAYOUT = {
    "positive_images": "positive/images", "positive_annotations": "positive/annotations",
    "crops": "positive/crops", "boxed_images": "positive/boxed_images",
    "negative_images": "negative/images", "negative_annotations": "negative/annotations",
}
PALETTE = ("#FF0000", "#00B050", "#0070FF", "#FFAA00", "#C000FF", "#00B8D4", "#FF1493", "#808000")


def is_derived(path: Path) -> bool:
    return bool(re.search(r"_crop_\d+_.+|_boxed$", path.stem, re.IGNORECASE))


def read_annotation(path: Path, width: int, height: int) -> tuple[dict, list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
        raise ValueError(f"不是有效的 LabelMe shapes 标注: {path}")
    json.dumps(document, allow_nan=False)
    for key, actual in (("imageWidth", width), ("imageHeight", height)):
        if key in document and document[key] != actual:
            raise ValueError(f"标注尺寸与图片不一致: {path}: {key}")
    objects = []
    for index, shape in enumerate(document["shapes"], 1):
        if not isinstance(shape, dict) or not isinstance(shape.get("label"), str) or not shape["label"].strip():
            raise ValueError(f"标注类别为空或无效: {path}, shape {index}")
        kind = shape.get("shape_type", "polygon")
        points = shape.get("points")
        if kind not in {"rectangle", "polygon"}:
            raise ValueError(f"只支持 rectangle/polygon，不能忽略其它类型: {path}, shape {index}: {kind}")
        if (not isinstance(points, list) or (kind == "rectangle" and len(points) != 2)
                or (kind == "polygon" and len(points) < 3)
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise ValueError(f"标注点无效: {path}, shape {index}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for p in points for v in p):
            raise ValueError(f"标注坐标必须是有限数字: {path}, shape {index}")
        x1, x2 = min(p[0] for p in points), max(p[0] for p in points)
        y1, y2 = min(p[1] for p in points), max(p[1] for p in points)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"标注框越界或为空: {path}, shape {index}")
        objects.append({"label": shape["label"], "shape_index": index, "shape_type": kind,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "crop_xyxy": [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]})
    return document, objects


def class_metadata(labels: set[str]) -> dict[str, dict]:
    slugs = {}
    for label in labels:
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", label)
        name = re.sub(r"\s+", "_", name).strip(" ._")[:60] or "class"
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", name.split(".")[0], re.IGNORECASE):
            name = "class_" + name
        slugs[label] = name
    frequency = Counter(name.casefold() for name in slugs.values())
    classes, used_colors = {}, set()
    for index, label in enumerate(sorted(labels, key=lambda value: (value.casefold(), value))):
        slug = slugs[label]
        if frequency[slug.casefold()] > 1:
            slug += "_" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
        if index < len(PALETTE):
            color = PALETTE[index]
        else:
            candidate = index
            while True:
                rgb = colorsys.hsv_to_rgb((candidate * 0.61803398875) % 1, 0.8, 0.95)
                color = "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)
                if color not in used_colors:
                    break
                candidate += 1
        used_colors.add(color)
        classes[label] = {"color": color, "crop_directory": slug}
    return classes


def inspect_source(source: Path, recursive: bool, include_derived: bool,
                   selected_paths: list[Path] | None = None) -> tuple[list[dict], list[str]]:
    if selected_paths is None:
        images = sorted(p for p in (source.rglob("*") if recursive else source.iterdir())
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        images = sorted(source / path for path in selected_paths)
        for image in images:
            if not image.resolve().is_relative_to(source) or not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"抽样图片必须是输入目录中的有效图片: {image}")
    entries, skipped, destinations = [], [], set()
    for image in images:
        relative = image.relative_to(source)
        if not include_derived and is_derived(image):
            skipped.append(relative.as_posix())
            continue
        key = relative.with_suffix(".json").as_posix().casefold()
        if key in destinations:
            raise ValueError(f"多张图片具有相同标注文件名，请先重命名: {relative}")
        destinations.add(key)
        annotation = image.with_suffix(".json")
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
        document, objects = read_annotation(annotation, width, height) if annotation.is_file() else (None, [])
        entries.append({"source": image, "relative": relative, "annotation": annotation if document is not None else None,
                        "document": document, "objects": objects, "width": width, "height": height,
                        "split": "positive" if objects else "negative",
                        "reason": "annotated" if objects else "empty_annotation" if document is not None else "missing_annotation"})
    if not entries:
        raise ValueError("没有可处理的原始图片（已默认排除派生裁剪图和带框图）")
    return entries, skipped


def flat_image_names(entries: list[dict]) -> dict[Path, Path]:
    """Keep original basenames; use a stable suffix only for stem collisions."""
    counts = Counter(entry["relative"].stem.casefold() for entry in entries)
    used = {entry["relative"].stem.casefold() for entry in entries if counts[entry["relative"].stem.casefold()] == 1}
    names = {}
    for entry in sorted(entries, key=lambda value: value["relative"].as_posix()):
        relative = entry["relative"]
        stem = relative.stem
        if counts[stem.casefold()] > 1:
            base = stem + "__" + hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:10]
            stem, number = base, 1
            while stem.casefold() in used:
                stem = f"{base}_{number}"
                number += 1
        used.add(stem.casefold())
        names[relative] = Path(stem + relative.suffix)
    return names


def draw_boxes(image: Image.Image, objects: list[dict], classes: dict, font, line_width: int) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for obj in objects:
        color = classes[obj["label"]]["color"]
        box = obj["bbox_xyxy"]
        draw.rectangle(box, outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), obj["label"], font=font)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        padding = 2
        x = max(0, min(int(box[0]), image.width - text_width - 2 * padding))
        y = max(0, int(box[1]) - text_height - 2 * padding)
        draw.rectangle([x, y, x + text_width + 2 * padding, y + text_height + 2 * padding], fill=color)
        draw.text((x + padding - text_box[0], y + padding - text_box[1]), obj["label"], fill="white", font=font)
    return preview


def prepare_dataset(source: Path, output: Path | None = None, *, recursive: bool = False,
                    include_derived: bool = False, dry_run: bool = False, line_width: int = 5,
                    font_size: int = 20, font_path: Path | None = None,
                    selected_paths: list[Path] | None = None) -> dict:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"输入目录不存在: {source}")
    if output is None and not source.name:
        raise ValueError("输入为磁盘根目录时必须指定 --output-dir")
    output = output.expanduser().resolve() if output is not None else source.with_name(source.name + "_prepared").resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("输出目录必须位于输入目录之外，不能覆盖源文件")
    if line_width < 1 or font_size < 1:
        raise ValueError("line_width 和 font_size 必须大于 0")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"输出目录非空，不覆盖已有结果，请指定新的 --output-dir: {output}")
    # Finish validation of every image and annotation before writing outputs.
    entries, skipped = inspect_source(source, recursive, include_derived, selected_paths)
    output_names = flat_image_names(entries)
    classes = class_metadata({obj["label"] for e in entries for obj in e["objects"]})
    object_counts = Counter(obj["label"] for e in entries for obj in e["objects"])
    for label, info in classes.items():
        info.update(object_count=object_counts[label],
                    image_count=sum(any(obj["label"] == label for obj in e["objects"]) for e in entries))
    report = {
        "schema": SCHEMA, "source_dir": str(source), "output_dir": str(output), "layout": LAYOUT,
        "recursive": recursive, "path_layout": "flat", "dry_run": dry_run, "image_count": len(entries),
        "positive_image_count": sum(e["split"] == "positive" for e in entries),
        "negative_image_count": sum(e["split"] == "negative" for e in entries),
        "crop_count": sum(len(e["objects"]) for e in entries), "classes": classes,
        "skipped_derived_images": skipped, "negative_policy": "missing or empty same-stem LabelMe annotation",
        "images": [],
    }
    if dry_run:
        return report
    font = load_font(font_path, font_size)
    output.mkdir(parents=True, exist_ok=True)
    for name in LAYOUT.values():
        (output / name).mkdir(parents=True, exist_ok=True)
    for entry in entries:
        relative, split = entry["relative"], entry["split"]
        name = output_names[relative]
        original = output / LAYOUT[split + "_images"] / name
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["source"], original)
        row = {"source_image": relative.as_posix(), "split": split, "reason": entry["reason"],
               "width": entry["width"], "height": entry["height"], "image": original.relative_to(output).as_posix(),
               "source_image_sha256": hashlib.sha256(entry["source"].read_bytes()).hexdigest(), "crops": []}
        if entry["document"] is not None:
            destination = output / LAYOUT[split + "_annotations"] / name.with_suffix(".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            document = dict(entry["document"])
            document.update(imagePath=os.path.relpath(original, destination.parent).replace("\\", "/"),
                            imageData=None, imageWidth=entry["width"], imageHeight=entry["height"])
            destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            row.update(annotation=destination.relative_to(output).as_posix(),
                       source_annotation=entry["annotation"].relative_to(source).as_posix(),
                       source_annotation_sha256=hashlib.sha256(entry["annotation"].read_bytes()).hexdigest())
        if split == "positive":
            with Image.open(entry["source"]) as opened:
                image = opened.convert("RGBA" if "A" in opened.getbands() or "transparency" in opened.info else "RGB")
            boxed = output / LAYOUT["boxed_images"] / (name.stem + "_boxed.png")
            boxed.parent.mkdir(parents=True, exist_ok=True)
            draw_boxes(image, entry["objects"], classes, font, line_width).save(boxed, format="PNG")
            row["boxed_image"] = boxed.relative_to(output).as_posix()
            for obj in entry["objects"]:
                slug = classes[obj["label"]]["crop_directory"]
                crop = (output / LAYOUT["crops"] / slug /
                        f"{name.stem}_crop_{obj['shape_index']:03d}_{slug}.png")
                crop.parent.mkdir(parents=True, exist_ok=True)
                image.crop(obj["crop_xyxy"]).save(crop, format="PNG")
                row["crops"].append({**obj, "image": crop.relative_to(output).as_posix()})
        report["images"].append(row)
    (output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整理 LabelMe 参考数据：正负样本、原图、JSON、分类裁剪图、多类别彩色标注全图")
    parser.add_argument("--input-dir", type=Path, required=True, help="原始图片及同名 LabelMe JSON 所在目录")
    parser.add_argument("--output-dir", type=Path, help="默认输出到同级 <原目录名>_prepared；不覆盖非空目录")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入子目录；输出不保留来源子目录")
    parser.add_argument("--include-derived", action="store_true", help="包括文件名含 _crop_序号_ 或以 _boxed 结尾的派生图片")
    parser.add_argument("--dry-run", action="store_true", help="只校验并显示数量/类别，不创建输出")
    parser.add_argument("--line-width", type=int, default=5)
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--font", type=Path, help="类别文字字体，可指定支持中文的 ttf/ttc")
    args = parser.parse_args(argv)
    try:
        report = prepare_dataset(args.input_dir, args.output_dir, recursive=args.recursive,
                                 include_derived=args.include_derived, dry_run=args.dry_run,
                                 line_width=args.line_width, font_size=args.font_size, font_path=args.font)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"数据整理失败: {exc}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
