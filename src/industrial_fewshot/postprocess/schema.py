"""Shared MLLM protocol; no model initialization at import time."""
from __future__ import annotations
import io
import json
import math
import mimetypes
import re
from pathlib import Path
from typing import Any

ANNOTATION_STATUSES = {"annotated", "no_target", "unable_to_determine"}
REASON_CODES = {
    "target_detected",
    "no_target",
    "blurry",
    "low_resolution",
    "too_dark",
    "overexposed",
    "backlit",
    "occluded",
    "target_too_small",
    "other",
}
UNABLE_REASON_CODES = REASON_CODES - {"target_detected", "no_target"}


def build_json_schema(config: dict[str, Any]) -> dict[str, Any]:
    label_names = [str(item["name"]) for item in config["labels"]]
    object_properties: dict[str, Any] = {
        "label": {"type": "string", "enum": label_names},
        "bbox_2d": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            },
            "minItems": 4,
            "maxItems": 4,
        },
    }
    object_required = ["label", "bbox_2d"]
    if config.get('require_box_reason'):
        object_properties['reason'] = {'type': 'string', 'minLength': 1,
            'description': '用一两句话说明可见的目标特征和该框的覆盖范围。'}
        object_required.append('reason')
    if config.get("require_confidence"):
        object_properties["confidence"] = {
            "type": "number", "minimum": 0, "maximum": 1,
        }
        object_required.append("confidence")
    # 只有显式开启的专用程序才会要求模型对每个框声明"框内确实有已离地悬吊的吊物"，
    # 关闭时 schema 与旧版本完全一致，不影响 batch_annotate/diaogou 等调用方。
    if config.get("require_suspension_evidence"):
        object_properties["hanging_object_confirmed"] = {"type": "boolean"}
        object_required.append("hanging_object_confirmed")
    return {
        "type": "object",
        "properties": {
            "annotation_status": {
                "type": "string",
                "enum": sorted(ANNOTATION_STATUSES),
            },
            "reason_code": {
                "type": "string",
                "enum": sorted(REASON_CODES),
            },
            "reason": {"type": "string", "minLength": 1},
            "target_count": {"type": "integer", "minimum": 0},
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": object_properties,
                    "required": object_required,
                    "additionalProperties": False,
                },
            }
        },
        "required": [
            "annotation_status",
            "target_count",
            "reason_code",
            "reason",
            "objects",
        ],
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
            "name": "image_annotation",
            "schema": build_json_schema(config),
        },
    }


def clean_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型响应中没有可解析的 JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型响应 JSON 顶层必须是 object")
    return value


def validate_prediction(
    prediction: dict[str, Any],
    label_names: set[str],
    width: int,
    height: int,
) -> dict[str, Any]:
    objects = prediction.get("objects")
    if not isinstance(objects, list):
        raise ValueError("模型响应缺少 objects 数组")

    annotation_status = prediction.get("annotation_status")
    reason_code = prediction.get("reason_code")
    reason = prediction.get("reason")
    target_count = prediction.get("target_count")
    if annotation_status not in ANNOTATION_STATUSES:
        raise ValueError(f"annotation_status 无效: {annotation_status!r}")
    if reason_code not in REASON_CODES:
        raise ValueError(f"reason_code 无效: {reason_code!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason 必须是非空字符串")
    if isinstance(target_count, bool) or not isinstance(target_count, int):
        raise ValueError("target_count 必须是非负整数")
    if target_count < 0:
        raise ValueError("target_count 必须是非负整数")
    if target_count != len(objects):
        raise ValueError(
            f"target_count={target_count} 与 objects 数量 {len(objects)} 不一致"
        )

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{index}] 不是 object")
        label = item.get("label")
        if label not in label_names:
            raise ValueError(f"objects[{index}].label 不在允许标签中: {label!r}")
        bbox = item.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"objects[{index}].bbox_2d 必须包含 4 个坐标")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            raise ValueError(f"objects[{index}].bbox_2d 包含非数字坐标")
        normalized = [int(round(float(value))) for value in bbox]
        x1, y1, x2, y2 = normalized
        if not all(0 <= value <= 1000 for value in normalized):
            raise ValueError(f"objects[{index}].bbox_2d 超出 0~1000: {normalized}")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"objects[{index}].bbox_2d 坐标顺序无效: {normalized}")

        pixel_bbox = [
            round(x1 / 1000 * width),
            round(y1 / 1000 * height),
            round(x2 / 1000 * width),
            round(y2 / 1000 * height),
        ]
        validated.append(
            {
                "label": label,
                "bbox_2d": normalized,
                "bbox_pixels_xyxy": pixel_bbox,
            }
        )
        if "confidence" in item:
            score = item["confidence"]
            if (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1
            ):
                raise ValueError(f"objects[{index}].confidence 必须是 0~1 的有限数字")
            validated[-1]["confidence"] = float(score)
        if 'reason' in item:
            if not isinstance(item['reason'], str) or not item['reason'].strip():
                raise ValueError(f'objects[{index}].reason 必须是非空字符串')
            validated[-1]['reason'] = item['reason'].strip()
    if validated:
        if annotation_status != "annotated" or reason_code != "target_detected":
            raise ValueError(
                "objects 非空时 annotation_status 必须为 annotated，"
                "reason_code 必须为 target_detected"
            )
    elif annotation_status == "annotated":
        raise ValueError("objects 为空时 annotation_status 不能为 annotated")
    elif annotation_status == "no_target" and reason_code != "no_target":
        raise ValueError("annotation_status=no_target 时 reason_code 必须为 no_target")
    elif annotation_status == "unable_to_determine" and reason_code not in UNABLE_REASON_CODES:
        raise ValueError(
            "annotation_status=unable_to_determine 时 reason_code 必须是图像质量、"
            "遮挡、目标过小或 other"
        )

    return {
        "annotation_status": annotation_status,
        "target_count": target_count,
        "reason_code": reason_code,
        "reason": reason.strip(),
        "objects": validated,
    }

