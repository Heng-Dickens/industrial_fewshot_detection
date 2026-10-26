#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 VLM 复核并纠正现有 LabelMe 矩形框标注。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import io
import json
import logging
import mimetypes
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

try:
    from industrial_fewshot.legacy.batch_annotate import clean_json_response, config_bool, config_float, read_completion
except ImportError:
    from industrial_fewshot.legacy.batch_annotate import clean_json_response, config_bool, config_float, read_completion

LOGGER = logging.getLogger("review_existing_annotations")
SCRIPT_DIR = Path(__file__).resolve().parent
# 通用程序放在 tool/ 下，配置统一集中在上一层的 config/ 目录
from industrial_fewshot.paths import LEGACY_CONFIG_DIR, resolve_location
CONFIG_DIR = LEGACY_CONFIG_DIR
DEFAULT_CONFIG_NAME = "Chache_cannotation_review_config.json"
DECISIONS = {"keep", "remove", "relabel", "uncertain"}

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 VLM 复核 LabelMe：优先删除误标/纠正错类，其次发现漏标。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / DEFAULT_CONFIG_NAME,
        help=f"配置文件地址，默认使用 configs/specific/{DEFAULT_CONFIG_NAME}",
    )
    parser.add_argument("--limit", type=int, help="最多处理多少张，建议先用 10")
    parser.add_argument("--concurrency", type=int, help="覆盖配置中的并发数")
    parser.add_argument("--force", action="store_true", help="重新复核成功记录")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不调用模型或写结果")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="同时检查 shapes 为空的图片；建议第二阶段使用",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON 不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return value


def resolve_path(base: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return resolve_location(path if path.is_absolute() else base / path)


def resolve_reference_path(
    config_base: Path, reference_dir: Path | None, raw: str | Path
) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return resolve_location(path)
    if reference_dir is not None:
        return resolve_location(reference_dir / path)
    return resolve_location(config_base / path)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_location(path)
    config = read_json(config_path)
    base = config_path.parent
    required = ["base_url", "model", "dataset_dir", "output_dir", "audit_dir", "labels"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段: {', '.join(missing)}")

    labels = config["labels"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels 必须是非空数组")
    names: list[str] = []
    for index, item in enumerate(labels):
        if not isinstance(item, dict):
            raise ValueError(f"labels[{index}] 必须是 object")
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not description:
            raise ValueError(f"labels[{index}] 必须包含非空 name 和 description")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("labels 中的标签名不能重复")

    dataset_dir = resolve_path(base, config["dataset_dir"])
    output_dir = resolve_path(base, config["output_dir"])
    audit_dir = resolve_path(base, config["audit_dir"])
    if output_dir == dataset_dir or audit_dir == dataset_dir:
        raise ValueError("output_dir/audit_dir 不能等于原始 dataset_dir")
    if is_within(output_dir, dataset_dir) or is_within(audit_dir, dataset_dir):
        raise ValueError("output_dir/audit_dir 不能位于原始 dataset_dir 内部")
    if not dataset_dir.is_dir():
        raise ValueError(f"dataset_dir 不存在: {dataset_dir}")

    config["config_path"] = str(config_path)
    config["dataset_dir"] = str(dataset_dir)
    config["output_dir"] = str(output_dir)
    config["audit_dir"] = str(audit_dir)
    config["checkpoint_jsonl"] = str(
        resolve_path(audit_dir, config.get("checkpoint_jsonl", "review.checkpoint.jsonl"))
    )
    config["label_names"] = names
    raw_reference_dir = config.get("reference_image_dir")
    reference_dir = (
        resolve_path(base, raw_reference_dir)
        if raw_reference_dir
        else None
    )
    explicit_reference_map = config.get("reference_images", {})
    if explicit_reference_map is None:
        explicit_reference_map = {}
    if not isinstance(explicit_reference_map, dict):
        raise ValueError("reference_images 必须是按标签名映射的 object")
    reference_paths: list[str | None] = []
    for item in labels:
        label_name = str(item["name"])
        raw_reference = item.get("reference_image")
        if raw_reference is None:
            raw_reference = explicit_reference_map.get(label_name)
        reference_path: Path | None = None
        if raw_reference:
            reference_path = resolve_reference_path(base, reference_dir, raw_reference)
        elif reference_dir is not None:
            for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                candidate = (reference_dir / f"{label_name}{suffix}").resolve()
                if candidate.is_file():
                    reference_path = candidate
                    break
        reference_paths.append(str(reference_path) if reference_path else None)
    use_reference_images = config_bool(
        config,
        "use_reference_images",
        bool(reference_dir is not None or any(reference_paths)),
    )
    config["reference_image_dir"] = str(reference_dir) if reference_dir else ""
    config["reference_image_paths"] = reference_paths
    config["use_reference_images"] = use_reference_images
    config["reference_image_manifest"] = []
    for key, default in (
        ("label_definitions_confirmed", False),
        ("review_empty_annotations", False),
        ("apply_remove", True),
        ("apply_relabel", True),
        ("apply_missing", False),
        ("save_review_images", True),
        ("stream", False),
        ("enable_thinking", False),
    ):
        config[key] = config_bool(config, key, default)
    config["temperature"] = config_float(config, "temperature", 0.0, 0.0, 2.0)
    config["top_p"] = config_float(config, "top_p", 0.9, 0.0, 1.0, False)
    config["missing_max_iou"] = config_float(config, "missing_max_iou", 0.25, 0.0, 1.0)
    config["concurrency"] = int(config.get("concurrency", 2))
    config["max_attempts"] = int(config.get("max_attempts", 3))
    config["max_tokens"] = int(config.get("max_tokens", 2048))
    config["timeout_seconds"] = float(config.get("timeout_seconds", 180))
    config["retry_delay_seconds"] = float(config.get("retry_delay_seconds", 2))
    config["structured_output_mode"] = str(config.get("structured_output_mode", "json_schema"))
    config["image_mode"] = str(config.get("image_mode", "hardlink"))
    config["prompt_version"] = str(config.get("prompt_version", "existing-label-review-v1"))
    if config["concurrency"] < 1 or config["max_attempts"] < 1 or config["max_tokens"] < 1:
        raise ValueError("concurrency、max_attempts 和 max_tokens 必须大于 0")
    if config["structured_output_mode"] not in {"json_schema", "json_object", "none"}:
        raise ValueError("structured_output_mode 只能是 json_schema/json_object/none")
    if config["image_mode"] not in {"hardlink", "copy", "none"}:
        raise ValueError("image_mode 只能是 hardlink/copy/none")
    return config


def resolve_image_path(json_path: Path, document: dict[str, Any]) -> Path:
    raw = str(document.get("imagePath", "")).strip()
    if raw:
        candidate = Path(raw)
        candidate = candidate if candidate.is_absolute() else json_path.parent / candidate
        if candidate.is_file():
            return candidate.resolve()
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = json_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"找不到标注对应图片: {json_path}")


def discover_annotations(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.rglob("*.json") if path.is_file())


def extract_candidates(document: dict[str, Any], label_names: set[str]) -> list[dict[str, Any]]:
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("LabelMe JSON 缺少 shapes 数组")
    candidates: list[dict[str, Any]] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"shapes[{index}] 不是 object")
        if shape.get("shape_type", "rectangle") != "rectangle":
            raise ValueError(f"shapes[{index}] 不是 rectangle")
        label = shape.get("label")
        if label not in label_names:
            raise ValueError(f"shapes[{index}] 包含未知标签: {label!r}")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError(f"shapes[{index}].points 必须包含两个点")
        try:
            x1, y1 = (float(value) for value in points[0])
            x2, y2 = (float(value) for value in points[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"shapes[{index}].points 坐标无效") from exc
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"shapes[{index}] 是空框")
        candidates.append({
            "object_id": f"obj_{index + 1:04d}",
            "shape_index": index,
            "label": str(label),
            "bbox_pixels_xyxy": [x1, y1, x2, y2],
        })
    return candidates


def build_prompt(config: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    label_lines = "\n".join(
        f'- `{item["name"]}`: {item["description"]}' for item in config["labels"]
    )
    existing = [
        {
            "object_id": item["object_id"],
            "current_label": item["label"],
            "bbox_pixels_xyxy": [round(v, 2) for v in item["bbox_pixels_xyxy"]],
        }
        for item in candidates
    ]
    reference_instruction = ""
    if config.get("use_reference_images", False):
        reference_instruction = (
            "\n后续消息会先提供每个类别各自的“类别参考图”，每张参考图前的文字就是该图的唯一类别。"
            "你必须先阅读全部类别参考图，提取每类稳定的车身、门架、叉臂、驾驶位和用途特征，"
            "再审核待标注图片。参考图只用于理解类别，不是待审核图片，也不能把参考图中的目标加入输出。"
            "禁止仅凭颜色、大小、是否有顶棚或当前已有标签判断类别，必须优先依据目标本体结构。"
            "对于 Chache1~Chache5：只有清晰看到该类别对应的叉车门架、立柱或叉臂结构，才允许 keep/relabel 为叉车类别。"
            "如果目标前部被裁掉、被遮挡或看不清，不能推断存在叉臂，应使用 uncertain。"
            "对于 Qianyinche：其关键排除条件是没有叉车门架和成对货叉；若前部清晰可见但没有叉车结构，"
            "应优先判为 Qianyinche，而不是任意一个 Chache 类别。\n"
        )
    custom_instruction = str(
        config.get("review_prompt", config.get("prompt", config.get("extra_instruction", "")))
    ).strip()
    prompt = f"""你是工业场景目标检测标注的质量审核员。首要任务是删除小模型产生的误标、背景框、重复框和明显错类；第二任务才是补充非常明确的漏标。

允许的 {len(config["labels"])} 类目标及现场定义：
{label_lines}

输入说明：参考图（如果提供）位于本消息最前面；之后是无标记原图和审核图。
审核图包含整图编号框和各编号框的局部裁图。参考图中的目标不是当前图片的候选框。
当前已有候选框：
{json.dumps(existing, ensure_ascii=False)}

请对每一个已有 object_id 恰好返回一次：
- keep：框内确实是 current_label 对应目标。程序保留原坐标，你不要重画。
- remove：不是允许目标、只框到背景/工人/模具/料架/护栏/固定设备，或与其他框重复。
- relabel：确实是允许目标之一，但类别错误；corrected_label 填正确标签。
- uncertain：遮挡、模糊、过小或类别依据不足，无法可靠判断。不要猜测。

审核规则：
1. 已有标签只是待审候选，不是事实，不能因为框存在就默认正确。
2. 必须依据目标本体结构，不能只按颜色、附近人员或背景位置判断。
3. remove 必须有明确视觉理由；拿不准用 uncertain。uncertain 默认保留并进入人工复核。
4. relabel 的 corrected_label 必须是另一个允许标签；其他决定必须填空字符串。
5. 不得遗漏、重复或编造已有 object_id。
6. missing_objects 只填整图中清晰、明显属于允许类别且未被现有框覆盖的目标；疑似目标不要新增。
7. 漏标框使用 0~1000 相对坐标 [x1,y1,x2,y2]，紧贴目标，左上小于右下。
8. 必须输出完整 JSON，不要输出 Markdown、代码块或额外解释。
"""
    prompt += reference_instruction
    if custom_instruction:
        prompt += f"\n用户配置的补充审核要求（必须同时遵守）：\n{custom_instruction}\n"
    return prompt


def build_json_schema(label_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "existing_reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string"},
                        "decision": {"type": "string", "enum": sorted(DECISIONS)},
                        "corrected_label": {"type": "string", "enum": [""] + label_names},
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["object_id", "decision", "corrected_label", "reason"],
                    "additionalProperties": False,
                },
            },
            "missing_objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": label_names},
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["label", "bbox_2d", "reason"],
                    "additionalProperties": False,
                },
            },
            "image_reason": {"type": "string", "minLength": 1},
        },
        "required": ["existing_reviews", "missing_objects", "image_reason"],
        "additionalProperties": False,
    }


def build_response_format(config: dict[str, Any]) -> dict[str, Any] | None:
    mode = config["structured_output_mode"]
    if mode == "none":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "existing_annotation_review",
            "schema": build_json_schema(config["label_names"]),
        },
    }


def encode_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def image_mime_type(path: Path, image_format: str | None) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    return Image.MIME.get(image_format or "", "image/jpeg")


def prepare_reference_images(config: dict[str, Any]) -> list[dict[str, Any]]:
    """读取每个类别的示例图，并生成发送给 VLM 的数据 URL。"""
    if not config.get("use_reference_images", False):
        config["_reference_images"] = []
        config["reference_image_manifest"] = []
        return []

    paths = config.get("reference_image_paths", [])
    if not isinstance(paths, list) or len(paths) != len(config["labels"]):
        raise ValueError("reference_image_paths 必须和 labels 数量一致")

    references: list[dict[str, Any]] = []
    missing: list[str] = []
    for label, raw_path in zip(config["labels"], paths):
        label_name = str(label["name"])
        if not raw_path:
            missing.append(f"{label_name}: 未配置 reference_image")
            continue
        path = Path(raw_path)
        if not path.is_file():
            missing.append(f"{label_name}: {path}")
            continue
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                image.verify()
        except Exception as exc:
            raise ValueError(f"类别参考图无法读取: {path}: {exc}") from exc
        mime_type = image_mime_type(path, image_format)
        references.append({
            "label": label_name,
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "mime_type": mime_type,
            "data_url": encode_data_url(data, mime_type),
        })
    if missing:
        raise ValueError(
            "启用了参考图，但以下类别缺少有效示例图: " + "; ".join(missing)
        )
    config["_reference_images"] = references
    config["reference_image_manifest"] = [
        {
            "label": item["label"],
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "mime_type": item["mime_type"],
        }
        for item in references
    ]
    return references


def build_review_board(image: Image.Image, candidates: list[dict[str, Any]]) -> bytes:
    """生成整图编号框 + 各框局部裁图，供 VLM 按 object_id 审核。"""
    source = image.convert("RGB")
    annotated = source.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    colors = ("#ff2d2d", "#00a8ff", "#ff9f1a", "#2ecc71", "#b05cff")
    for index, item in enumerate(candidates):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = item["bbox_pixels_xyxy"]
        width = max(3, round(max(source.size) / 600))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = f'{item["object_id"]} {item["label"]}'
        box = draw.textbbox((x1, y1), text, font=font, stroke_width=1)
        draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=color)
        draw.text((x1, y1), text, fill="white", font=font, stroke_width=1, stroke_fill=color)

    full = annotated.copy()
    full.thumbnail((1400, 900))
    panel_width, panel_height, columns = 340, 280, 4
    rows = (len(candidates) + columns - 1) // columns
    board_width = max(1400, panel_width * columns)
    board_height = full.height + 50 + rows * panel_height
    board = Image.new("RGB", (board_width, max(board_height, full.height + 50)), "white")
    board.paste(full, ((board_width - full.width) // 2, 40))
    ImageDraw.Draw(board).text((10, 10), "FULL IMAGE WITH NUMBERED BOXES", fill="black", font=font)

    for index, item in enumerate(candidates):
        x1, y1, x2, y2 = item["bbox_pixels_xyxy"]
        width, height = x2 - x1, y2 - y1
        pad_x, pad_y = max(width * 0.25, 30), max(height * 0.25, 30)
        crop_box = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(source.width, int(x2 + pad_x)),
            min(source.height, int(y2 + pad_y)),
        )
        crop = annotated.crop(crop_box)
        crop.thumbnail((panel_width - 20, panel_height - 45))
        left = (index % columns) * panel_width
        top = full.height + 50 + (index // columns) * panel_height
        board.paste(crop, (left + (panel_width - crop.width) // 2, top + 30))
        ImageDraw.Draw(board).text(
            (left + 8, top + 8),
            f'{item["object_id"]} / {item["label"]}',
            fill=colors[index % len(colors)],
            font=font,
        )

    buffer = io.BytesIO()
    board.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def validate_review(
    value: dict[str, Any], candidates: list[dict[str, Any]], label_names: set[str]
) -> dict[str, Any]:
    reviews = value.get("existing_reviews")
    missing = value.get("missing_objects")
    image_reason = value.get("image_reason")
    if not isinstance(reviews, list) or not isinstance(missing, list):
        raise ValueError("响应缺少 existing_reviews 或 missing_objects 数组")
    if not isinstance(image_reason, str) or not image_reason.strip():
        raise ValueError("image_reason 必须是非空字符串")

    expected = {item["object_id"]: item for item in candidates}
    validated_reviews: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            raise ValueError(f"existing_reviews[{index}] 不是 object")
        object_id = review.get("object_id")
        decision = review.get("decision")
        corrected = review.get("corrected_label")
        reason = review.get("reason")
        if object_id not in expected or object_id in seen:
            raise ValueError(f"无效或重复 object_id: {object_id!r}")
        if decision not in DECISIONS:
            raise ValueError(f"{object_id} 的 decision 无效: {decision!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{object_id} 的 reason 不能为空")
        if decision == "relabel":
            if corrected not in label_names or corrected == expected[object_id]["label"]:
                raise ValueError(f"{object_id} 的 corrected_label 无效: {corrected!r}")
        elif corrected != "":
            raise ValueError(f"{object_id} 非 relabel 时 corrected_label 必须为空")
        seen.add(object_id)
        validated_reviews.append({
            "object_id": object_id,
            "decision": decision,
            "corrected_label": corrected,
            "reason": reason.strip(),
        })
    if seen != set(expected):
        raise ValueError(f"模型未完整审核全部 object_id: {sorted(set(expected) - seen)}")

    validated_missing: list[dict[str, Any]] = []
    for index, item in enumerate(missing):
        if not isinstance(item, dict) or item.get("label") not in label_names:
            raise ValueError(f"missing_objects[{index}] 标签无效")
        bbox = item.get("bbox_2d")
        reason = item.get("reason")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"missing_objects[{index}].bbox_2d 无效")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
            raise ValueError(f"missing_objects[{index}].bbox_2d 包含非数字")
        coords = [int(round(float(v))) for v in bbox]
        if not all(0 <= v <= 1000 for v in coords) or coords[0] >= coords[2] or coords[1] >= coords[3]:
            raise ValueError(f"missing_objects[{index}].bbox_2d 越界或顺序无效")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"missing_objects[{index}].reason 不能为空")
        validated_missing.append({"label": item["label"], "bbox_2d": coords, "reason": reason.strip()})
    return {
        "existing_reviews": validated_reviews,
        "missing_objects": validated_missing,
        "image_reason": image_reason.strip(),
    }


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def apply_review(
    document: dict[str, Any],
    candidates: list[dict[str, Any]],
    review: dict[str, Any],
    width: int,
    height: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按保守策略生成新 LabelMe；永不修改传入的原文档。"""
    result = copy.deepcopy(document)
    shapes = document["shapes"]
    reviews = {item["object_id"]: item for item in review["existing_reviews"]}
    corrected_shapes: list[dict[str, Any]] = []
    final_boxes: list[list[float]] = []
    changes: list[dict[str, Any]] = []

    for candidate in candidates:
        item = reviews[candidate["object_id"]]
        shape = copy.deepcopy(shapes[candidate["shape_index"]])
        decision = item["decision"]
        applied = "kept"
        if decision == "remove" and config["apply_remove"]:
            applied = "removed"
        else:
            if decision == "relabel" and config["apply_relabel"]:
                shape["label"] = item["corrected_label"]
                applied = "relabeled"
            elif decision == "remove":
                applied = "remove_suggested_not_applied"
            elif decision == "relabel":
                applied = "relabel_suggested_not_applied"
            elif decision == "uncertain":
                applied = "kept_for_manual_review"
            corrected_shapes.append(shape)
            final_boxes.append(candidate["bbox_pixels_xyxy"])
        changes.append({
            "object_id": candidate["object_id"],
            "original_label": candidate["label"],
            "decision": decision,
            "corrected_label": item["corrected_label"],
            "applied_action": applied,
            "reason": item["reason"],
        })

    for index, missing in enumerate(review["missing_objects"], start=1):
        x1, y1, x2, y2 = missing["bbox_2d"]
        pixel_box = [
            round(x1 / 1000 * width, 2),
            round(y1 / 1000 * height, 2),
            round(x2 / 1000 * width, 2),
            round(y2 / 1000 * height, 2),
        ]
        max_iou = max((box_iou(pixel_box, box) for box in final_boxes), default=0.0)
        applied = "missing_suggested_not_applied"
        if max_iou > config["missing_max_iou"]:
            applied = "missing_rejected_as_overlap"
        elif config["apply_missing"]:
            corrected_shapes.append({
                "label": missing["label"],
                "points": [[pixel_box[0], pixel_box[1]], [pixel_box[2], pixel_box[3]]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            })
            final_boxes.append(pixel_box)
            applied = "missing_added"
        changes.append({
            "object_id": f"missing_{index:04d}",
            "original_label": None,
            "decision": "add_missing",
            "corrected_label": missing["label"],
            "bbox_2d": missing["bbox_2d"],
            "bbox_pixels_xyxy": pixel_box,
            "max_iou_with_kept": round(max_iou, 4),
            "applied_action": applied,
            "reason": missing["reason"],
        })

    result["shapes"] = corrected_shapes
    return result, changes


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    if mode == "none":
        return "none"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def config_signature(config: dict[str, Any]) -> str:
    keys = (
        "model", "prompt_version", "labels", "temperature", "top_p", "max_tokens",
        "enable_thinking", "stream", "apply_remove", "apply_relabel", "apply_missing",
        "missing_max_iou", "structured_output_mode", "review_empty_annotations",
        "use_reference_images", "reference_image_manifest", "review_prompt",
        "prompt", "extra_instruction",
    )
    payload = {key: config.get(key) for key in keys}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_successful(checkpoint: Path, signature: str) -> set[str]:
    successful: set[str] = set()
    if not checkpoint.is_file():
        return successful
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "success" and record.get("config_signature") == signature:
            successful.add(str(record.get("relative_json", "")))
    return successful


class CheckpointWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()

    async def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()


async def request_review(
    client: Any,
    config: dict[str, Any],
    prompt: str,
    original_url: str,
    review_url: str,
    candidates: list[dict[str, Any]],
    reference_images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reference_images = reference_images or []
    if reference_images:
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "下面先提供类别参考图。每张参考图前的文字标签与紧随其后的图片一一对应；"
                "请先阅读全部参考图，再处理后面的待审核图片。"
            ),
        }]
        for index, reference in enumerate(reference_images, start=1):
            content.append({
                "type": "text",
                "text": f"类别参考图 {index}/{len(reference_images)}：{reference['label']}",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": reference["data_url"]},
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = [{"type": "text", "text": prompt}]
    content.extend([
        {"type": "image_url", "image_url": {"url": original_url}},
        {"type": "image_url", "image_url": {"url": review_url}},
    ])
    request: dict[str, Any] = {
        "model": config["model"],
        "messages": [{
            "role": "user",
            "content": content,
        }],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "stream": config["stream"],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]}
        },
    }
    if config["stream"]:
        request["stream_options"] = {"include_usage": True}
    response_format = build_response_format(config)
    if response_format is not None:
        request["response_format"] = response_format
    response = await client.chat.completions.create(**request)
    content, finish_reason, usage, reasoning = await read_completion(response, config["stream"])
    if not content.strip():
        raise ValueError(
            f"模型响应为空（finish_reason={finish_reason!r}, reasoning字符数={len(reasoning or '')}）"
        )
    value = clean_json_response(content)
    return {
        "review": validate_review(value, candidates, set(config["label_names"])),
        "raw_response": content,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def write_corrected_pair(
    document: dict[str, Any],
    source_image: Path,
    relative_json: Path,
    config: dict[str, Any],
) -> tuple[Path, str]:
    output_dir = Path(config["output_dir"])
    destination_json = output_dir / relative_json
    destination_image = destination_json.parent / source_image.name
    action = materialize_image(source_image, destination_image, config["image_mode"])
    result = copy.deepcopy(document)
    if config["image_mode"] == "none":
        result["imagePath"] = str(source_image)
    else:
        result["imagePath"] = source_image.name
    atomic_write_json(destination_json, result)
    return destination_json, action


def basic_record(
    json_path: Path,
    image_path: Path,
    relative_json: Path,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relative_json": relative_json.as_posix(),
        "source_json": str(json_path),
        "source_image": str(image_path),
        "source_json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "model": config["model"],
        "prompt_version": config["prompt_version"],
        "config_signature": config_signature(config),
        "existing_objects": candidates,
        "reference_images": config.get("reference_image_manifest", []),
    }


async def process_one(
    client: Any,
    writer: CheckpointWriter,
    json_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_dir = Path(config["dataset_dir"])
    relative_json = json_path.relative_to(dataset_dir)
    document = await asyncio.to_thread(read_json, json_path)
    image_path = resolve_image_path(json_path, document)
    candidates = extract_candidates(document, set(config["label_names"]))
    record = basic_record(json_path, image_path, relative_json, candidates, config)
    prompt = build_prompt(config, candidates)
    record["prompt"] = prompt

    image_bytes = await asyncio.to_thread(image_path.read_bytes)
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        image_format = opened.format
        board_bytes = build_review_board(image, candidates)
    original_url = encode_data_url(image_bytes, image_mime_type(image_path, image_format))
    review_url = encode_data_url(board_bytes, "image/jpeg")

    errors: list[str] = []
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            response = await request_review(
                client,
                config,
                prompt,
                original_url,
                review_url,
                candidates,
                config.get("_reference_images", []),
            )
            corrected, changes = apply_review(
                document, candidates, response["review"], width, height, config
            )
            corrected["imageWidth"] = width
            corrected["imageHeight"] = height
            destination_json, image_action = await asyncio.to_thread(
                write_corrected_pair,
                corrected,
                image_path,
                relative_json,
                config,
            )
            record.update({
                "status": "success",
                "attempts": attempt,
                "output_json": str(destination_json),
                "image_action": image_action,
                "review": response["review"],
                "changes": changes,
                "change_counts": dict(Counter(item["applied_action"] for item in changes)),
                "raw_response": response["raw_response"],
                "reasoning_content": response["reasoning_content"],
                "finish_reason": response["finish_reason"],
                "usage": response["usage"],
            })
            break
        except Exception as exc:  # 单图失败必须保留审计和原标注
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < config["max_attempts"]:
                await asyncio.sleep(config["retry_delay_seconds"] * (2 ** (attempt - 1)))
    else:
        destination_json, image_action = await asyncio.to_thread(
            write_corrected_pair, document, image_path, relative_json, config
        )
        record.update({
            "status": "failed_passthrough",
            "attempts": config["max_attempts"],
            "output_json": str(destination_json),
            "image_action": image_action,
            "errors": errors,
            "changes": [],
            "change_counts": {},
        })

    if config["save_review_images"]:
        board_path = Path(config["audit_dir"]) / "review_images" / relative_json.with_suffix(".jpg")
        board_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(board_path.write_bytes, board_bytes)
        record["review_image"] = str(board_path)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    record["completed_at"] = now_iso()
    audit_path = Path(config["audit_dir"]) / "records" / relative_json
    await asyncio.to_thread(atomic_write_json, audit_path, record)
    await writer.append(record)
    return record


def inspect_dataset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(config["dataset_dir"])
    label_names = set(config["label_names"])
    items: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    empty_count = 0
    errors: list[str] = []
    for json_path in discover_annotations(dataset_dir):
        try:
            document = read_json(json_path)
            image_path = resolve_image_path(json_path, document)
            candidates = extract_candidates(document, label_names)
            label_counts.update(item["label"] for item in candidates)
            if not candidates:
                empty_count += 1
            items.append({
                "json_path": json_path,
                "image_path": image_path,
                "relative_json": json_path.relative_to(dataset_dir),
                "candidate_count": len(candidates),
            })
        except Exception as exc:
            errors.append(f"{json_path}: {type(exc).__name__}: {exc}")
    stats = {
        "annotation_files": len(items),
        "existing_objects": sum(item["candidate_count"] for item in items),
        "empty_annotations": empty_count,
        "label_counts": dict(sorted(label_counts.items())),
        "invalid_items": len(errors),
        "errors": errors[:20],
    }
    return items, stats


def sync_passthrough_empty(items: list[dict[str, Any]], config: dict[str, Any]) -> int:
    """完整运行时把未送审的空标注同步到新数据集，确保负样本不丢失。"""
    copied = 0
    for item in items:
        if item["candidate_count"] != 0:
            continue
        destination = Path(config["output_dir"]) / item["relative_json"]
        if destination.exists():
            continue
        document = read_json(item["json_path"])
        write_corrected_pair(
            document,
            item["image_path"],
            item["relative_json"],
            config,
        )
        copied += 1
    return copied


def dry_run_report(
    config: dict[str, Any],
    stats: dict[str, Any],
    eligible: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "dataset_dir": config["dataset_dir"],
        "output_dir": config["output_dir"],
        "audit_dir": config["audit_dir"],
        "model": config["model"],
        "label_definitions_confirmed": config["label_definitions_confirmed"],
        "labels": config["labels"],
        "use_reference_images": config["use_reference_images"],
        "reference_images": config.get("reference_image_manifest", []),
        "policy": {
            "apply_remove": config["apply_remove"],
            "apply_relabel": config["apply_relabel"],
            "apply_missing": config["apply_missing"],
            "review_empty_annotations": config["review_empty_annotations"],
        },
        "dataset": stats,
        "eligible_for_vlm": len(eligible),
        "scheduled": len(pending),
        "sample_json": [item["relative_json"].as_posix() for item in pending[:10]],
    }


async def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.concurrency is not None:
        if args.concurrency < 1:
            raise ValueError("--concurrency 必须大于 0")
        config["concurrency"] = args.concurrency
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit 不能小于 0")
    if args.include_empty:
        config["review_empty_annotations"] = True

    prepare_reference_images(config)

    items, stats = await asyncio.to_thread(inspect_dataset, config)
    if stats["invalid_items"]:
        raise ValueError(
            f"发现 {stats['invalid_items']} 个无效图片/标注对，示例: {stats['errors'][0]}"
        )
    eligible = [
        item for item in items
        if item["candidate_count"] > 0 or config["review_empty_annotations"]
    ]
    signature = config_signature(config)
    successful = load_successful(Path(config["checkpoint_jsonl"]), signature)
    pending = [
        item for item in eligible
        if args.force or item["relative_json"].as_posix() not in successful
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    if args.dry_run:
        print(json.dumps(dry_run_report(config, stats, eligible, pending), ensure_ascii=False, indent=2))
        return 0
    if not config["label_definitions_confirmed"]:
        raise ValueError(
            "类别定义尚未确认：请先修改 labels.description，确认所有类别的业务差异，"
            "再将 label_definitions_confirmed 设为 true"
        )

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["audit_dir"]).mkdir(parents=True, exist_ok=True)
    passthrough_empty = 0
    if args.limit is None and not config["review_empty_annotations"]:
        passthrough_empty = await asyncio.to_thread(sync_passthrough_empty, items, config)

    records: list[dict[str, Any]] = []
    if pending:
        from industrial_fewshot.dependencies import load_openai_client
        AsyncOpenAI = load_openai_client()
        api_key_env = str(config.get("api_key_env", "VLLM_API_KEY"))
        api_key = os.getenv(api_key_env, str(config.get("api_key", "EMPTY")))
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=config["base_url"],
            timeout=config["timeout_seconds"],
            max_retries=0,
        )
        writer = CheckpointWriter(Path(config["checkpoint_jsonl"]))
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        for item in pending:
            queue.put_nowait(item)
        worker_count = min(config["concurrency"], len(pending))
        for _ in range(worker_count):
            queue.put_nowait(None)

        async def worker(worker_id: int) -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                try:
                    LOGGER.info("worker-%d 复核 %s", worker_id, item["relative_json"])
                    record = await process_one(client, writer, item["json_path"], config)
                    records.append(record)
                    LOGGER.info("worker-%d 完成 %s: %s", worker_id, item["relative_json"], record["status"])
                except Exception as exc:
                    LOGGER.exception("worker-%d 未预期失败 %s", worker_id, item["relative_json"])
                    failure = {
                        "relative_json": item["relative_json"].as_posix(),
                        "source_json": str(item["json_path"]),
                        "status": "worker_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "config_signature": signature,
                        "completed_at": now_iso(),
                    }
                    records.append(failure)
                    await writer.append(failure)
                finally:
                    queue.task_done()

        try:
            await asyncio.gather(*(worker(i + 1) for i in range(worker_count)))
        finally:
            await client.close()

    status_counts = Counter(record["status"] for record in records)
    change_counts: Counter[str] = Counter()
    for record in records:
        change_counts.update(record.get("change_counts", {}))
    summary = {
        "generated_at": now_iso(),
        "dataset_dir": config["dataset_dir"],
        "output_dir": config["output_dir"],
        "audit_dir": config["audit_dir"],
        "config_signature": signature,
        "dataset": stats,
        "eligible_for_vlm": len(eligible),
        "already_successful": len(successful),
        "scheduled_this_run": len(pending),
        "passthrough_empty_annotations": passthrough_empty,
        "status_counts": dict(status_counts),
        "change_counts": dict(change_counts),
        "use_reference_images": config["use_reference_images"],
        "reference_images": config.get("reference_image_manifest", []),
    }
    await asyncio.to_thread(
        atomic_write_json, Path(config["audit_dir"]) / "run_summary.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if status_counts.get("failed_passthrough") or status_counts.get("worker_error") else 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        args = parse_args()
        config = load_config(args.config)
        return asyncio.run(run(args, config))
    except KeyboardInterrupt:
        LOGGER.warning("用户中断；已完成记录保留在检查点中")
        return 130
    except Exception as exc:
        LOGGER.error("执行失败: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
