"""Shared visual references: annotated full images, positive crops and negatives."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from industrial_fewshot.data.labelme import read_json
from industrial_fewshot.paths import stable_reference_identity

LOGGER = logging.getLogger(__name__)
REFERENCE_IMAGE_MAX_SIDE = 1280
REFERENCE_MODES = ("text_only", "text_json", "text_crops", "text_json_crops")


def _reference_directory(root: Path, kind: str) -> Path:
    """Read prepared datasets without mixing positive derivatives into negatives."""
    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            if read_json(manifest).get("schema") == "annotationtool.reference_dataset.v1":
                locations = {"json": "positive/annotations", "crops": "positive/crops", "negatives": "negative/images"}
                return root / locations[kind]
        except (OSError, ValueError):
            pass
    return root


def configure_reference_inputs(config: dict, mode: str) -> None:
    """Apply a named input combination, preserving the configured sample caps."""
    if mode not in ("configured", *REFERENCE_MODES):
        raise ValueError(f"未知 reference_mode: {mode}; 可选 {REFERENCE_MODES}")
    keys = ("max_reference_examples", "max_reference_crops", "max_reference_negatives")
    limits = config.setdefault("reference_limits", {key: config[key] for key in keys})
    config.update(limits)
    config["reference_mode"] = mode
    if mode == "configured":
        return
    use_json = mode in {"text_json", "text_json_crops"}
    use_crops = mode in {"text_crops", "text_json_crops"}
    config["max_reference_examples"] = limits["max_reference_examples"] if use_json else 0
    config["max_reference_crops"] = limits["max_reference_crops"] if use_crops else 0
    config["max_reference_negatives"] = 0
    if use_json and config["max_reference_examples"] == 0:
        raise ValueError(f"{mode} 需要 max_reference_examples > 0")
    if use_crops and config["max_reference_crops"] == 0:
        raise ValueError(f"{mode} 需要 max_reference_crops > 0")


def discover_json_references(reference_dir: Path, limit: int) -> list[dict]:
    """Read LabelMe text alone; corresponding reference images are unnecessary."""
    if limit <= 0:
        return []
    examples = []
    for path in sorted(_reference_directory(reference_dir, "json").rglob("*.json")):
        try:
            document = read_json(path)
            if not isinstance(document.get("shapes"), list):
                continue
            if any(isinstance(document.get(key), bool) or not isinstance(document.get(key), (int, float))
                   or document[key] <= 0 for key in ("imageWidth", "imageHeight")):
                continue
            examples.append({"json_path": str(path), "compact_json": compact_labelme_document(document)})
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            LOGGER.warning("参考 JSON 不可用，跳过: %s: %s", path, exc)
            continue
        if len(examples) >= limit:
            break
    return examples


def _prepare_named_inputs(config: dict) -> None:
    examples, crops = [], []
    if config["max_reference_examples"] or config["max_reference_crops"]:
        if not config.get("reference_image_dir"):
            raise ValueError("当前 reference_mode 需要 reference_image_dir")
        directory = Path(config["reference_image_dir"])
        if not directory.is_dir():
            raise ValueError(f"参考目录不存在: {directory}")
        if config["max_reference_examples"]:
            examples = discover_json_references(directory, config["max_reference_examples"])
            if not examples:
                raise ValueError(f"{config['reference_mode']} 没有可用的 LabelMe 参考 JSON: {directory}")
        for crop in discover_crop_references(directory, config["max_reference_crops"]):
            rendered, mime = render_reference_image(crop["image_path"], [])
            crops.append({"image_path": str(crop["image_path"]), "image_sha256": crop["image_sha256"],
                          "rendered_data_url": encode_data_url(rendered, mime)})
        if config["max_reference_crops"] and not crops:
            raise ValueError(f"{config['reference_mode']} 没有可用的正例裁剪小图: {directory}")
    config.update(reference_examples=examples, reference_crops=crops, reference_negatives=[])
    manifest = {
        "reference_mode": config["reference_mode"], "json_examples": examples,
        "crops": [{k: c[k] for k in ("image_path", "image_sha256")} for c in crops],
    }
    config["reference_manifest_sha256"] = hashlib.sha256(
        json.dumps(stable_reference_identity(manifest), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def discover_reference_pairs(
    reference_dir: Path, max_examples: int
) -> list[dict[str, Any]]:
    """扫描参考目录中的 LabelMe JSON，与对应图片配对，按文件名排序后截断。"""
    pairs: list[dict[str, Any]] = []
    if max_examples <= 0:
        return pairs
    for json_path in sorted(_reference_directory(reference_dir, "json").rglob("*.json")):
        try:
            document = read_json(json_path)
        except ValueError as exc:
            LOGGER.warning("参考标注 JSON 无法读取，跳过: %s: %s", json_path, exc)
            continue
        shapes = document.get("shapes")
        if not isinstance(shapes, list) or "imagePath" not in document:
            LOGGER.warning("不是有效的 LabelMe 标注文件，跳过: %s", json_path)
            continue

        raw = str(document.get("imagePath", "")).strip()
        image_path: Path | None = None
        if raw:
            candidate = Path(raw)
            candidate = (
                candidate if candidate.is_absolute() else json_path.parent / candidate
            )
            if candidate.is_file():
                image_path = candidate.resolve()
        if image_path is None:
            for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                candidate = json_path.with_suffix(suffix)
                if candidate.is_file():
                    image_path = candidate.resolve()
                    break
        if image_path is None:
            LOGGER.warning("参考标注缺少对应图片，跳过: %s", json_path)
            continue

        pairs.append(
            {
                "json_path": json_path,
                "image_path": image_path,
                "document": document,
                "shapes": shapes,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        )
        if len(pairs) >= max_examples:
            break
    return pairs



def encode_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"



def compact_labelme_document(document: dict[str, Any]) -> str:
    """生成发给模型的紧凑标准标注 JSON 文本：去掉 imageData、坐标保留两位小数。"""
    payload = {key: value for key, value in document.items() if key != "imageData"}
    if isinstance(payload.get("shapes"), list):
        shapes = []
        for shape in payload["shapes"]:
            if not isinstance(shape, dict):
                shapes.append(shape)
                continue
            copied = dict(shape)
            points = copied.get("points")
            if isinstance(points, list):
                copied["points"] = [
                    [round(float(x), 2), round(float(y), 2)] for x, y in points
                ]
            shapes.append(copied)
        payload["shapes"] = shapes
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))



def is_reference_crop(image_path: Path) -> bool:
    """识别本项目导出的正例裁剪文件：原图名_crop_序号_类别。"""
    return re.search(r"_crop_\d+_.+", image_path.stem, re.IGNORECASE) is not None



def discover_crop_references(
    reference_dir: Path, max_examples: int
) -> list[dict[str, Any]]:
    """读取已有正例小图，不裁剪或改写参考目录中的文件。"""
    if max_examples <= 0:
        return []
    crops: list[dict[str, Any]] = []
    for image_path in sorted(_reference_directory(reference_dir, "crops").rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and is_reference_crop(image_path)
        ):
            crops.append({
                "image_path": image_path.resolve(),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            })
            if len(crops) >= max_examples:
                break
    return crops



def discover_negative_references(
    reference_dir: Path, max_examples: int
) -> list[dict[str, Any]]:
    """扫描参考目录里“没有同名 LabelMe JSON”的图片，作为不符合标注条件的负例。

    正例裁剪图无论是否启用发送，都不能被当成负例。其余不带同名 .json 的图片是负例。
    """
    negatives: list[dict[str, Any]] = []
    if max_examples <= 0:
        return negatives
    for image_path in sorted(_reference_directory(reference_dir, "negatives").rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if is_reference_crop(image_path):
            continue
        if image_path.with_suffix(".json").is_file():
            continue  # 有标注的是正例，由 discover_reference_pairs 负责
        negatives.append(
            {
                "image_path": image_path.resolve(),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        )
        if len(negatives) >= max_examples:
            break
    return negatives



def render_reference_image(
    image_path: Path, shapes: list[dict[str, Any]]
) -> tuple[bytes, str]:
    """把标准标注框画到参考图上并缩小，返回 JPEG 字节和 MIME 类型。"""
    with Image.open(image_path) as source_image:
        rendered = source_image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    color = "#ff2d2d"
    line_width = max(3, round(max(rendered.size) / 600))
    for shape in shapes:
        points = shape.get("points")
        if shape.get("shape_type") != "rectangle" or not isinstance(points, list):
            continue
        if len(points) < 2:
            continue
        if not all(
            isinstance(point, (list, tuple)) and len(point) >= 2 for point in points[:2]
        ):
            continue
        x1 = float(points[0][0])
        y1 = float(points[0][1])
        x2 = float(points[1][0])
        y2 = float(points[1][1])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = str(shape.get("label", ""))
        box = draw.textbbox((x1, y1), text, font=font, stroke_width=1)
        draw.rectangle(
            (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=color
        )
        draw.text(
            (x1, y1), text, fill="white", font=font, stroke_width=1, stroke_fill=color
        )
    rendered.thumbnail((REFERENCE_IMAGE_MAX_SIDE, REFERENCE_IMAGE_MAX_SIDE))
    buffer = io.BytesIO()
    rendered.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), "image/jpeg"



def prepare_reference_examples(config: dict[str, Any]) -> None:
    """准备带框正例、正例细节小图、负例，并计算完整输入的复用指纹。"""
    if config.get('reference_mode') in {'pet_dino', 'wedetect'}:
        from industrial_fewshot.detectors.prompt_native import prepare_references
        prepare_references(config)
        return
    if config.get('reference_mode') == 'yoloe_fusion':
        from industrial_fewshot.detectors.yoloe_fusion import prepare_references
        prepare_references(config)
        return
    if config.get('reference_mode') == 'yoloe_text':
        from industrial_fewshot.detectors.yoloe_text import prepare_references
        prepare_references(config)
        return
    if config.get('reference_mode') == 'supervised_onnx':
        from industrial_fewshot.detectors.yolo_onnx import prepare_references
        prepare_references(config)
        return
    if config.get("reference_mode") == "ovd_text":
        from industrial_fewshot.detectors.groundingdino_text import prepare_references
        prepare_references(config)
        return
    if config.get("reference_mode") == "yoloe_visual":
        from industrial_fewshot.detectors.yoloe_visual import prepare_references
        prepare_references(config)
        return
    if config.get("reference_mode", "configured") != "configured":
        _prepare_named_inputs(config)
        return
    reference_dir = Path(config.get("reference_image_dir") or ".")
    max_examples = config["max_reference_examples"]
    pairs = discover_reference_pairs(reference_dir, max_examples)
    if max_examples > 0 and not pairs:
        raise ValueError(
            f"参考目录中没有可用的“图片 + 标注 JSON”示例对: {reference_dir}"
        )
    examples: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            rendered, rendered_mime = render_reference_image(
                pair["image_path"], pair["shapes"]
            )
        except Exception as exc:
            raise ValueError(
                f"参考图渲染失败: {pair['image_path']}: {exc}"
            ) from exc
        examples.append(
            {
                "json_path": str(pair["json_path"]),
                "image_path": str(pair["image_path"]),
                "image_sha256": pair["image_sha256"],
                "rendered_data_url": encode_data_url(rendered, rendered_mime),
                "compact_json": compact_labelme_document(pair["document"]),
            }
        )
    config["reference_examples"] = examples

    crops: list[dict[str, Any]] = []
    for crop in discover_crop_references(
        reference_dir, config.get("max_reference_crops", 6)
    ):
        try:
            rendered, rendered_mime = render_reference_image(crop["image_path"], [])
        except Exception as exc:
            raise ValueError(f"正例裁剪图读取失败: {crop['image_path']}: {exc}") from exc
        crops.append({
            "image_path": str(crop["image_path"]),
            "image_sha256": crop["image_sha256"],
            "rendered_data_url": encode_data_url(rendered, rendered_mime),
        })
    config["reference_crops"] = crops

    negatives: list[dict[str, Any]] = []
    for negative in discover_negative_references(
        reference_dir, config["max_reference_negatives"]
    ):
        try:
            rendered, rendered_mime = render_reference_image(
                negative["image_path"], []
            )
        except Exception as exc:
            raise ValueError(
                f"负例参考图渲染失败: {negative['image_path']}: {exc}"
            ) from exc
        negatives.append(
            {
                "image_path": str(negative["image_path"]),
                "image_sha256": negative["image_sha256"],
                "rendered_data_url": encode_data_url(rendered, rendered_mime),
            }
        )
    config["reference_negatives"] = negatives

    manifest = [
        {
            "json_path": str(pair["json_path"]),
            "image_sha256": pair["image_sha256"],
            "shapes": pair["shapes"],
        }
        for pair in pairs
    ] + [
        {
            "image_path": item["image_path"],
            "image_sha256": item["image_sha256"],
            "shapes": [],
        }
        for item in negatives
    ] + [
        {
            "kind": "positive_crop",
            "image_path": item["image_path"],
            "image_sha256": item["image_sha256"],
        }
        for item in crops
    ]
    config["reference_manifest_sha256"] = hashlib.sha256(
        json.dumps(stable_reference_identity(manifest), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()



def build_messages(
    config: dict[str, Any],
    prompt: str,
    reference_examples: list[dict[str, Any]],
    current_url: str,
) -> list[dict[str, Any]]:
    """构造多图消息：带框正例、正例小图、负例在前，当前待标全图始终在最后。"""
    if config.get("reference_mode") == "composed":
        from industrial_fewshot.prompts.visual import build_suite_messages
        return build_suite_messages(config, prompt, current_url)
    if config.get("reference_mode", "configured") != "configured":
        content = []
        for index, example in enumerate(reference_examples, 1):
            content.append({"type": "text", "text": (
                f"参考标注 JSON {index}/{len(reference_examples)}（仅提供文本，不附参考原图）：\n"
                + example["compact_json"]
                + "\n这是其它参考图片的 LabelMe 示例，不是当前图片的真值，也不是本次要求的输出格式。"
                "示例坐标属于参考图，不能复制为当前全图的坐标。"
            )})
        for index, crop in enumerate(config.get("reference_crops", []), 1):
            content.extend([
                {"type": "text", "text": f"正例目标裁剪图 {index}：学习目标外观与组件范围，随后在当前全图搜索。"},
                {"type": "image_url", "image_url": {"url": crop["rendered_data_url"]}},
            ])
        instruction = "只使用本次实际提供的参考信息；提示词提及但未提供的参考类型直接略过。\n" + prompt
        if content:
            instruction += "\n参考信息仅用于学习。最后一张是唯一待识别全图，请搜索全部区域，只返回该全图的目标和坐标。"
        content.extend([{"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": current_url}}])
        return content
    label_names = [str(item["name"]) for item in config["labels"]]
    crops = config.get("reference_crops") or []
    negatives = config.get("reference_negatives") or []
    if reference_examples or crops or negatives:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "下面先提供参考示例，分三类。"
                    "① 已标注示例：图上带红框、并附该图的标准 LabelMe 标注，"
                    "表示该图的目标符合标注条件、应该这样框；"
                    "那段 LabelMe JSON 只是给你看目标该框成什么样，不是你要输出的格式。"
                    "② 正例目标裁剪图：按人工标注框裁好的目标小图，用于学习目标外观和完整组件范围，"
                    "没有红框也属于正例。"
                    "③ 未标注示例：明确标记为负例的全图，表示该图不符合标注条件，"
                    "正确答案是不标注任何框。"
                    "预期类别："
                    + "、".join(label_names)
                    + "。请先看完全部示例，再处理最后一张当前图片。"
                ),
            }
        ]
        for index, example in enumerate(reference_examples, start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"已标注示例 {index}/{len(reference_examples)}"
                        "（红色框为标准标注结果）："
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": example["rendered_data_url"]},
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"示例 {index} 的标准 LabelMe 标注 JSON：\n"
                        f"{example['compact_json']}"
                    ),
                }
            )
        for index, crop in enumerate(crops, start=1):
            content.append({
                "type": "text",
                "text": (
                    f"正例目标裁剪图 {index}/{len(crops)}"
                    f"（类别：{'、'.join(label_names)}；已按人工标注框裁剪，"
                    "没有红框也属于正例，不是负例、不是待检测图片）："
                    "观察目标的外观、结构、部件关系及整体框选范围；"
                    "随后在最后一张待标注全图中搜索相同类别的目标。"
                ),
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": crop["rendered_data_url"]},
            })
        for index, negative in enumerate(negatives, start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"未标注示例 {index}/{len(negatives)}"
                        "（负例全图：这张图不符合标注条件，正确答案是不标注任何框）："
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": negative["rendered_data_url"]},
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    prompt
                    + "\n\n下面最后一张是唯一需要标注的待标注全图。参考示例只用于学习"
                    "什么该标、什么不该标，不得把参考图中的目标当作当前图片中的目标。"
                    "请搜索这张全图的所有区域，只返回这张图中的目标和相对这张全图的坐标。"
                ),
            }
        )
    else:
        content = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url", "image_url": {"url": current_url}})
    return content

