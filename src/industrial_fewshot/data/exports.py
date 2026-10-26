#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拆分 VLM 预标注为 LabelMe、Pascal VOC、YOLO、COCO 和模型回复。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

try:
    from industrial_fewshot.legacy.batch_annotate import append_dataset_suffix, config_bool, dataset_identifier
except ImportError:  # 直接执行 python annotate/annotation_tools.py
    from industrial_fewshot.legacy.batch_annotate import append_dataset_suffix, config_bool, dataset_identifier


LOGGER = logging.getLogger("annotation_tools")
SCRIPT_DIR = Path(__file__).resolve().parent
# 通用程序放在 tool/ 下，配置统一集中在上一层的 config/ 目录
from industrial_fewshot.paths import LEGACY_CONFIG_DIR, resolve_location
CONFIG_DIR = LEGACY_CONFIG_DIR
PALETTE = [
    "#00AEEF",
    "#F15A24",
    "#8DC63F",
    "#92278F",
    "#FFCB05",
    "#ED1C24",
    "#00A99D",
    "#3C5AA6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取批量标注 JSON，拆分为 LabelMe/VOC/YOLO 和模型回复。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="annotations.json 地址；未指定时读取配置中的 output_json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "annotation_config.json",
        help="批量标注配置文件地址",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="导出根目录；默认在标注文件旁创建带时间戳的新目录",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("labelme", "voc", "yolo", "replies", "preview", "coco"),
        default=("labelme", "voc", "yolo", "replies"),
        help="需要生成的格式，默认 labelme voc yolo replies",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="各数据集如何组织图片；hardlink 失败时自动复制",
    )
    parser.add_argument("--font", type=Path, help="预览图标签字体文件")
    parser.add_argument("--font-size", type=int, default=24, help="标签字号")
    parser.add_argument("--line-width", type=int, default=4, help="边界框线宽")
    parser.add_argument("--limit", type=int, help="最多转换多少条成功记录")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON 文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 文件格式错误: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return value


def resolve_from(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return resolve_location(path)


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input is not None:
        return resolve_location(args.input)
    config_path = resolve_location(args.config)
    config = read_json(config_path)
    output_json = config.get("output_json")
    if not output_json:
        raise ValueError(f"配置文件缺少 output_json: {config_path}")
    input_path = resolve_from(config_path.parent, str(output_json))
    if config_bool(config, "append_image_dir_suffix", False):
        image_dir = config.get("image_dir")
        if not image_dir:
            raise ValueError(
                f"启用 append_image_dir_suffix 时配置必须包含 image_dir: {config_path}"
            )
        identifier = dataset_identifier(
            resolve_from(config_path.parent, str(image_dir))
        )
        input_path = append_dataset_suffix(input_path, identifier)
    return input_path


def default_output_dir(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_path.parent / f"export_{timestamp}"


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"输出目录非空，为避免混入旧标注已停止: {path}。请指定新的 --output-dir。"
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("标注 JSON 缺少 annotations 数组")
    invalid = [index for index, item in enumerate(annotations) if not isinstance(item, dict)]
    if invalid:
        raise ValueError(f"annotations 中包含非 object 项，首个位置: {invalid[0]}")
    return annotations


def collect_label_names(
    document: dict[str, Any], annotations: Iterable[dict[str, Any]]
) -> list[str]:
    names: list[str] = []
    inference = document.get("inference")
    if isinstance(inference, dict):
        labels = inference.get("labels")
        if isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    name = item["name"]
                    if name and name not in names:
                        names.append(name)
    observed: set[str] = set()
    for record in annotations:
        objects = record.get("objects")
        if not isinstance(objects, list):
            continue
        for item in objects:
            if isinstance(item, dict) and isinstance(item.get("label"), str):
                observed.add(item["label"])
    names.extend(sorted(observed.difference(names)))
    if not names:
        raise ValueError("标注中没有可用的标签定义")
    return names


def successful_records(
    annotations: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    records = [item for item in annotations if item.get("status") == "success"]
    records.sort(key=lambda item: str(item.get("relative_path", "")))
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit 不能小于 0")
        records = records[:limit]
    return records


def image_root(document: dict[str, Any], input_path: Path) -> Path:
    dataset = document.get("dataset")
    if isinstance(dataset, dict) and dataset.get("image_root"):
        return resolve_from(input_path.parent, str(dataset["image_root"]))
    return input_path.parent


def record_relative_path(record: dict[str, Any]) -> Path:
    value = record.get("relative_path") or record.get("file_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("记录缺少 relative_path/file_name")
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"relative_path 不是安全相对路径: {value}")
    return path


def source_image_path(record: dict[str, Any], root: Path) -> Path:
    image_path = record.get("image_path")
    if isinstance(image_path, str):
        candidate = Path(image_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return (root / record_relative_path(record)).resolve()


def record_dimensions(record: dict[str, Any]) -> tuple[int, int]:
    width = record.get("width")
    height = record.get("height")
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("图片宽高无效")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("记录缺少图片 width/height")
    width_int, height_int = int(width), int(height)
    if width_int <= 0 or height_int <= 0:
        raise ValueError("图片宽高必须大于 0")
    return width_int, height_int


def object_pixel_box(
    record: dict[str, Any], annotation: dict[str, Any]
) -> tuple[float, float, float, float]:
    width, height = record_dimensions(record)
    pixel_box = annotation.get("bbox_pixels_xyxy")
    if isinstance(pixel_box, list) and len(pixel_box) == 4:
        values = pixel_box
    else:
        normalized = annotation.get("bbox_2d")
        if not isinstance(normalized, list) or len(normalized) != 4:
            raise ValueError("目标缺少 bbox_pixels_xyxy/bbox_2d")
        values = [
            float(normalized[0]) / 1000 * width,
            float(normalized[1]) / 1000 * height,
            float(normalized[2]) / 1000 * width,
            float(normalized[3]) / 1000 * height,
        ]
    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("边界框包含非数字坐标") from exc
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"边界框坐标顺序无效: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def objects_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    objects = record.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("记录中的 objects 不是数组")
    return [item for item in objects if isinstance(item, dict)]


def stable_color(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).digest()[0]
    return PALETTE[digest % len(PALETTE)]


def load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    if size <= 0:
        raise ValueError("--font-size 必须大于 0")
    candidates = []
    if font_path is not None:
        candidates.append(font_path.expanduser().resolve())
    candidates.extend(
        [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def save_preview_image(
    source: Path,
    destination: Path,
    record: dict[str, Any],
    font: ImageFont.ImageFont,
    line_width: int,
) -> tuple[int, int]:
    if line_width <= 0:
        raise ValueError("--line-width 必须大于 0")
    invalid_count = 0
    drawn_count = 0
    with Image.open(source) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for annotation in objects_of(record):
        label = str(annotation.get("label", "unknown"))
        try:
            x1, y1, x2, y2 = object_pixel_box(record, annotation)
        except ValueError as exc:
            invalid_count += 1
            LOGGER.warning("跳过无效框 %s: %s", source, exc)
            continue
        color = stable_color(label)
        box = [round(x1), round(y1), round(x2), round(y2)]
        draw.rectangle(box, outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        padding = max(2, line_width)
        text_x = box[0]
        text_y = max(0, box[1] - text_height - 2 * padding)
        draw.rectangle(
            [
                text_x,
                text_y,
                text_x + text_width + 2 * padding,
                text_y + text_height + 2 * padding,
            ],
            fill=color,
        )
        draw.text(
            (text_x + padding, text_y + padding - text_box[1]),
            label,
            fill="white",
            font=font,
        )
        drawn_count += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(destination, quality=95)
    else:
        image.save(destination)
    return drawn_count, invalid_count


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    if mode == "none":
        return "none"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        LOGGER.warning("硬链接失败，改为复制 %s: %s", source, exc)
        shutil.copy2(source, destination)
        return "copy_fallback"


def valid_labeled_objects(
    record: dict[str, Any], label_to_id: dict[str, int]
) -> Iterable[tuple[dict[str, Any], tuple[float, float, float, float], int]]:
    for annotation in objects_of(record):
        label = annotation.get("label")
        if label not in label_to_id:
            LOGGER.warning("跳过未知标签: %r", label)
            continue
        try:
            box = object_pixel_box(record, annotation)
        except ValueError as exc:
            LOGGER.warning("跳过无效框: %s", exc)
            continue
        yield annotation, box, label_to_id[str(label)]


def export_previews(
    records: list[dict[str, Any]],
    root: Path,
    output_dir: Path,
    font: ImageFont.ImageFont,
    line_width: int,
) -> dict[str, int]:
    stats = {"images": 0, "objects": 0, "invalid_objects": 0, "missing_images": 0}
    for record in records:
        source = source_image_path(record, root)
        if not source.is_file():
            stats["missing_images"] += 1
            LOGGER.warning("图片不存在，无法可视化: %s", source)
            continue
        relative = record_relative_path(record)
        drawn, invalid = save_preview_image(
            source, output_dir / relative, record, font, line_width
        )
        stats["images"] += 1
        stats["objects"] += drawn
        stats["invalid_objects"] += invalid
    return stats


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def export_labelme(
    records: list[dict[str, Any]],
    root: Path,
    output_dir: Path,
    label_names: list[str],
    image_mode: str,
) -> dict[str, int]:
    """逐图导出 LabelMe 5.1.1 rectangle JSON。"""
    label_to_id = {label: index for index, label in enumerate(label_names)}
    stats = {
        "images": 0,
        "annotation_files": 0,
        "objects": 0,
        "invalid_objects": 0,
        "missing_images": 0,
        "copy_fallbacks": 0,
    }
    used_json_paths: set[Path] = set()
    for record in records:
        source = source_image_path(record, root)
        if not source.is_file():
            stats["missing_images"] += 1
            LOGGER.warning("图片不存在，跳过 LabelMe 导出: %s", source)
            continue
        relative = record_relative_path(record)
        json_relative = relative.with_suffix(".json")
        if json_relative in used_json_paths:
            raise ValueError(f"多个图片会生成同一个 LabelMe JSON: {json_relative}")
        used_json_paths.add(json_relative)

        if image_mode != "none":
            action = materialize_image(source, output_dir / relative, image_mode)
            if action == "copy_fallback":
                stats["copy_fallbacks"] += 1
            image_path_value = source.name
        else:
            image_path_value = str(source)

        width, height = record_dimensions(record)
        valid_objects = list(valid_labeled_objects(record, label_to_id))
        shapes = []
        for annotation, (x1, y1, x2, y2), _ in valid_objects:
            shapes.append(
                {
                    "label": str(annotation["label"]),
                    'description': annotation.get('reason', ''),
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )
        labelme = {
            "version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path_value,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        destination = output_dir / json_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(labelme, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stats["images"] += 1
        stats["annotation_files"] += 1
        stats["objects"] += len(shapes)
        stats["invalid_objects"] += max(0, len(objects_of(record)) - len(shapes))
    return stats


def image_depth(record: dict[str, Any], source: Path) -> int:
    for key in ("depth", "image_depth", "channels"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    try:
        with Image.open(source) as image:
            return max(1, len(image.getbands()))
    except OSError as exc:
        LOGGER.warning("无法读取图片通道数，VOC depth 使用默认值3 %s: %s", source, exc)
        return 3


def add_xml_text(parent: ET.Element, tag: str, value: Any) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(value)
    return child


def export_voc(
    records: list[dict[str, Any]],
    root: Path,
    output_dir: Path,
    label_names: list[str],
    image_mode: str,
) -> dict[str, int]:
    """逐图导出 Pascal VOC XML。"""
    label_to_id = {label: index for index, label in enumerate(label_names)}
    stats = {
        "images": 0,
        "annotation_files": 0,
        "objects": 0,
        "invalid_objects": 0,
        "missing_images": 0,
        "copy_fallbacks": 0,
    }
    used_xml_paths: set[Path] = set()
    for record in records:
        source = source_image_path(record, root)
        if not source.is_file():
            stats["missing_images"] += 1
            LOGGER.warning("图片不存在，跳过 VOC 导出: %s", source)
            continue
        relative = record_relative_path(record)
        xml_relative = relative.with_suffix(".xml")
        if xml_relative in used_xml_paths:
            raise ValueError(f"多个图片会生成同一个 VOC XML: {xml_relative}")
        used_xml_paths.add(xml_relative)

        if image_mode != "none":
            action = materialize_image(
                source, output_dir / "JPEGImages" / relative, image_mode
            )
            if action == "copy_fallback":
                stats["copy_fallbacks"] += 1

        width, height = record_dimensions(record)
        depth = image_depth(record, source)
        annotation_root = ET.Element("annotation")
        folder = relative.parent.name if relative.parent.name else root.name
        add_xml_text(annotation_root, "folder", folder)
        add_xml_text(annotation_root, "filename", source.name)
        add_xml_text(annotation_root, "path", source)
        source_node = ET.SubElement(annotation_root, "source")
        add_xml_text(source_node, "database", "Unknown")
        size_node = ET.SubElement(annotation_root, "size")
        add_xml_text(size_node, "width", width)
        add_xml_text(size_node, "height", height)
        add_xml_text(size_node, "depth", depth)
        add_xml_text(annotation_root, "segmented", 0)

        valid_objects = list(valid_labeled_objects(record, label_to_id))
        for annotation, (x1, y1, x2, y2), _ in valid_objects:
            xmin, ymin, xmax, ymax = map(round, (x1, y1, x2, y2))
            object_node = ET.SubElement(annotation_root, "object")
            add_xml_text(object_node, "name", annotation["label"])
            add_xml_text(object_node, "pose", "Unspecified")
            truncated = int(xmin <= 0 or ymin <= 0 or xmax >= width or ymax >= height)
            add_xml_text(object_node, "truncated", truncated)
            add_xml_text(object_node, "difficult", 0)
            box_node = ET.SubElement(object_node, "bndbox")
            add_xml_text(box_node, "xmin", xmin)
            add_xml_text(box_node, "ymin", ymin)
            add_xml_text(box_node, "xmax", xmax)
            add_xml_text(box_node, "ymax", ymax)

        destination = output_dir / "Annotations" / xml_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(annotation_root)
        ET.indent(tree, space="  ")
        tree.write(destination, encoding="utf-8", xml_declaration=False)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        stats["images"] += 1
        stats["annotation_files"] += 1
        stats["objects"] += len(valid_objects)
        stats["invalid_objects"] += max(
            0, len(objects_of(record)) - len(valid_objects)
        )
    return stats


def export_model_replies(
    records: list[dict[str, Any]], output_dir: Path
) -> dict[str, int]:
    """按图片相对路径将模型回复和推理元数据写入独立 JSON。"""
    fields = (
        "image_id",
        "dataset_id",
        "relative_path",
        "image_path",
        "status",
        "model",
        "prompt_version",
        "generation_settings",
        "annotation_status",
        "target_count",
        "reason_code",
        "reason",
        "objects",
        "initial_annotation",
        'model_feedback_attempts',
        'initial_suspension_gate',
        'suspension_gate',
        'feedback_json',
        'feedback_text',
        'model_feedback_attempts',
        'initial_suspension_gate',
        'suspension_gate',
        'feedback_json',
        'feedback_text',
        "initial_raw_response",
        "initial_reasoning_content",
        "initial_finish_reason",
        "initial_usage",
        "review_attempted",
        "review_performed",
        "review_changed",
        "review_changed_target_count",
        "review_error",
        "raw_response",
        "reasoning_content",
        "finish_reason",
        "usage",
        "attempts",
        "completed_at",
        "latency_ms",
    )
    stats = {
        "files": 0,
        "initial_raw_responses": 0,
        "final_raw_responses": 0,
        "missing_final_raw_responses": 0,
    }
    used_paths: set[Path] = set()
    for record in records:
        reply = {field: record.get(field) for field in fields}
        relative_json = record_relative_path(record).with_suffix(".json")
        if relative_json in used_paths:
            raise ValueError(
                f"多个图片会生成同一个模型回复文件: {relative_json}"
            )
        used_paths.add(relative_json)
        destination = output_dir / relative_json
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(reply, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stats["files"] += 1
        if isinstance(reply.get("initial_raw_response"), str):
            stats["initial_raw_responses"] += 1
        if isinstance(reply.get("raw_response"), str):
            stats["final_raw_responses"] += 1
        else:
            stats["missing_final_raw_responses"] += 1
    return stats


def export_yolo(
    records: list[dict[str, Any]],
    root: Path,
    output_dir: Path,
    label_names: list[str],
    image_mode: str,
) -> dict[str, int]:
    label_to_id = {label: index for index, label in enumerate(label_names)}
    stats = {
        "images": 0,
        "objects": 0,
        "missing_images": 0,
        "copy_fallbacks": 0,
    }
    used_label_paths: set[Path] = set()
    for record in records:
        source = source_image_path(record, root)
        if not source.is_file():
            stats["missing_images"] += 1
            LOGGER.warning("图片不存在，跳过 YOLO 导出: %s", source)
            continue
        relative = record_relative_path(record)
        label_relative = relative.with_suffix(".txt")
        if label_relative in used_label_paths:
            raise ValueError(f"多个图片会生成同一个 YOLO 标签文件: {label_relative}")
        used_label_paths.add(label_relative)
        if image_mode != "none":
            action = materialize_image(
                source, output_dir / "images" / relative, image_mode
            )
            if action == "copy_fallback":
                stats["copy_fallbacks"] += 1

        width, height = record_dimensions(record)
        lines: list[str] = []
        for _, (x1, y1, x2, y2), class_id in valid_labeled_objects(
            record, label_to_id
        ):
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )
        label_path = output_dir / "labels" / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["objects"] += len(lines)

    (output_dir / "classes.txt").write_text(
        "\n".join(label_names) + "\n", encoding="utf-8"
    )
    yaml_lines = [
        f"path: {yaml_string(str(output_dir.resolve()))}",
        "train: images",
        "names:",
    ]
    yaml_lines.extend(
        f"  {index}: {yaml_string(label)}" for index, label in enumerate(label_names)
    )
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return stats


def export_coco(
    records: list[dict[str, Any]],
    root: Path,
    output_dir: Path,
    label_names: list[str],
    image_mode: str,
) -> dict[str, int]:
    label_to_id = {label: index + 1 for index, label in enumerate(label_names)}
    stats = {
        "images": 0,
        "objects": 0,
        "missing_images": 0,
        "copy_fallbacks": 0,
    }
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for record in records:
        source = source_image_path(record, root)
        if not source.is_file():
            stats["missing_images"] += 1
            LOGGER.warning("图片不存在，跳过 COCO 导出: %s", source)
            continue
        relative = record_relative_path(record)
        if image_mode != "none":
            action = materialize_image(
                source, output_dir / "images" / relative, image_mode
            )
            if action == "copy_fallback":
                stats["copy_fallbacks"] += 1
            file_name = f"images/{relative.as_posix()}"
        else:
            file_name = str(source)
        width, height = record_dimensions(record)
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        for _, (x1, y1, x2, y2), category_id in valid_labeled_objects(
            record, label_to_id
        ):
            box_width = x2 - x1
            box_height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        round(x1, 3),
                        round(y1, 3),
                        round(box_width, 3),
                        round(box_height, 3),
                    ],
                    "area": round(box_width * box_height, 3),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1
        stats["images"] += 1
    stats["objects"] = len(annotations)
    coco = {
        "info": {
            "description": "VLM-assisted annotations exported by annotation_tools.py",
            "date_created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_image_root": str(root),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": label, "supercategory": "object"}
            for index, label in enumerate(label_names)
        ],
    }
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / "instances.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> Path:
    input_path = resolve_input_path(args)
    document = read_json(input_path)
    annotations = validate_document(document)
    records = successful_records(annotations, args.limit)
    labels = collect_label_names(document, annotations)
    root = image_root(document, input_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(input_path).resolve()
    )
    prepare_output_dir(output_dir)
    formats = list(dict.fromkeys(args.formats))
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_annotation": str(input_path),
        "source_image_root": str(root),
        "output_dir": str(output_dir),
        "formats": formats,
        "image_mode": args.image_mode,
        "label_names": labels,
        "input_record_count": len(annotations),
        "successful_record_count": sum(
            item.get("status") == "success" for item in annotations
        ),
        "failed_records_skipped": sum(
            item.get("status") != "success" for item in annotations
        ),
        "source_no_annotation_reason_counts": dict(
            sorted(
                Counter(
                    str(item.get("reason_code", "missing"))
                    for item in annotations
                    if item.get("status") == "success" and not item.get("objects")
                ).items()
            )
        ),
        "selected_record_count": len(records),
        "results": {},
    }
    if "preview" in formats:
        font = load_font(args.font, args.font_size)
        report["results"]["preview"] = export_previews(
            records,
            root,
            output_dir / "preview",
            font,
            args.line_width,
        )
    if "labelme" in formats:
        report["results"]["labelme"] = export_labelme(
            records,
            root,
            output_dir / "labelme",
            labels,
            args.image_mode,
        )
    if "voc" in formats:
        report["results"]["voc"] = export_voc(
            records,
            root,
            output_dir / "voc",
            labels,
            args.image_mode,
        )
    if "yolo" in formats:
        report["results"]["yolo"] = export_yolo(
            records,
            root,
            output_dir / "yolo",
            labels,
            args.image_mode,
        )
    if "replies" in formats:
        report["results"]["replies"] = export_model_replies(
            records,
            output_dir / "model_replies",
        )
    if "coco" in formats:
        report["results"]["coco"] = export_coco(
            records,
            root,
            output_dir / "coco",
            labels,
            args.image_mode,
        )
    write_report(report, output_dir)
    return output_dir


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        output_dir = run(parse_args())
        LOGGER.info("转换完成: %s", output_dir)
        return 0
    except Exception as exc:
        LOGGER.error("转换失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
