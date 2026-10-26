"""LabelMe ground truth validation and isolated prediction JSON serialization."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

LABELME_VERSION = "6.3.1"
MISSING_GT_POLICIES = {"skip", "error", "negative"}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return value



def load_ground_truth(path: Path, width: int, height: int, labels: set[str]) -> list[dict]:
    document = read_json(path)
    if not isinstance(document.get("shapes"), list):
        raise ValueError(f"真值不是 LabelMe shapes 数组: {path}")
    for key, actual in (("imageWidth", width), ("imageHeight", height)):
        if key in document and document[key] != actual:
            raise ValueError(f"真值图像尺寸不匹配: {path}: {key}")
    objects = []
    for shape in document["shapes"]:
        if not isinstance(shape, dict):
            raise ValueError(f"真值 shape 无效: {path}")
        if shape.get("label") not in labels:
            continue
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"当前评估只接受 rectangle 真值: {path}")
        if any(shape.get("flags", {}).get(flag) for flag in ("ignore", "difficult", "iscrowd")):
            raise ValueError(f"当前评估不支持 ignore/difficult/crowd 标记: {path}")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) != 2 or any(not isinstance(p, list) or len(p) != 2 for p in points):
            raise ValueError(f"真值矩形坐标无效: {path}")
        values = [v for p in points for v in p]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"真值坐标必须是有限数字: {path}")
        x1, x2 = sorted([points[0][0], points[1][0]])
        y1, y2 = sorted([points[0][1], points[1][1]])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"真值框越界或为空: {path}")
        objects.append({"label": shape["label"], "bbox_pixels_xyxy": [x1, y1, x2, y2]})
    return objects


def load_evaluation_truth(path: Path, width: int, height: int, labels: set[str],
                          missing: str) -> tuple[list[dict] | None, str]:
    """Apply the dataset's declared meaning of a missing annotation file."""
    if missing not in MISSING_GT_POLICIES:
        raise ValueError("missing_ground_truth 只能是 skip、error 或 negative")
    if path.is_file():
        return load_ground_truth(path, width, height, labels), "json"
    if missing == "error":
        raise ValueError(f"缺少真值 JSON: {path}")
    return ([], "missing_as_negative") if missing == "negative" else (None, "missing_excluded")


def truth_summary(items: list[dict]) -> dict:
    return {
        "ground_truth_image_count": sum(i["ground_truth"] is not None for i in items),
        "ground_truth_json_image_count": sum(i.get("ground_truth_source") == "json" for i in items),
        "assumed_negative_image_count": sum(i.get("ground_truth_source") == "missing_as_negative" for i in items),
        "unlabeled_image_count": sum(i["ground_truth"] is None for i in items),
    }


def validate_pixel_prediction(prediction: dict, labels: set[str], width: int, height: int) -> dict:
    """Validate local detector outputs while preserving floating-point pixel boxes."""
    from industrial_fewshot.postprocess.schema import validate_prediction
    objects = prediction.get("objects")
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("本地检测结果 objects 必须是对象数组")
    # Reuse status/count/label/confidence validation without quantizing geometry.
    normalized = validate_prediction({**prediction, "objects": [
        {**obj, "bbox_2d": [0, 0, 1000, 1000]} for obj in objects]}, labels, width, height)
    validated = []
    for obj in objects:
        box = obj.get("bbox_pixels_xyxy")
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
            raise ValueError("本地检测像素框必须是 4 个有限数字")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("本地检测像素框越界或退化")
        if "confidence" not in obj:
            raise ValueError("本地检测结果缺少 confidence")
        validated.append({"label": obj["label"], "bbox_pixels_xyxy": list(box),
                          "bbox_2d": [x1 / width * 1000, y1 / height * 1000,
                                      x2 / width * 1000, y2 / height * 1000],
                          "confidence": float(obj["confidence"])})
    return {**normalized, "objects": validated}



def build_labelme_document(
    image_name: str,
    width: int,
    height: int,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    """把已验证的目标列表组装成与参考示例一致的标准 LabelMe JSON。"""
    shapes = []
    for item in objects:
        x1, y1, x2, y2 = item["bbox_pixels_xyxy"]
        shapes.append(
            {
                "label": str(item["label"]),
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": item.get('reason', ''),
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        )
    for shape, item in zip(shapes, objects):
        if "confidence" in item:
            shape["confidence"] = item["confidence"]
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }



def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

