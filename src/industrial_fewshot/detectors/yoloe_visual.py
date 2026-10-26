"""Local, cross-image YOLOE visual prompting; torch is imported only for inference."""
from __future__ import annotations

import asyncio
import hashlib
import io
import math
import os
import sys
import time
from pathlib import Path

from PIL import Image

from industrial_fewshot.data.labelme import read_json, validate_pixel_prediction
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.supports import ReferencePool

from industrial_fewshot.paths import PROJECT_ROOT


def yoloe_settings(raw: dict, base: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("yoloe 必须是配置对象")
    settings = {"source_dir": "../third_party/yoloe", "weights": "../weights/yoloe/yoloe-v8s-seg.pt",
                "positive_count": 5, "selection_seed": 0, "device": "cpu", "imgsz": 640,
                "confidence": 0.001, "nms_iou": 0.7, "max_detections": 100,
                "aggregation": "normalized_mean", "exclude_reference_overlap": True,
                "torch_threads": 4, **raw}
    allowed = {"source_dir", "weights", "positive_count", "selection_seed", "device", "imgsz",
               "confidence", "nms_iou", "max_detections", "aggregation", "exclude_reference_overlap", "torch_threads"}
    if set(settings) - allowed:
        raise ValueError(f"未知 yoloe 配置字段: {sorted(set(settings) - allowed)}")
    for key in ("positive_count", "imgsz", "max_detections", "torch_threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"yoloe.{key} 必须是正整数")
    if settings["imgsz"] % 32:
        raise ValueError("yoloe.imgsz 必须是 32 的倍数")
    if type(settings["selection_seed"]) is not int:
        raise ValueError("yoloe.selection_seed 必须是整数")
    for key in ("confidence", "nms_iou"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"yoloe.{key} 必须是 0~1 的有限数字")
    if not isinstance(settings["device"], str) or not settings["device"].strip():
        raise ValueError("yoloe.device 必须是 cpu、0 或 cuda:0 等设备字符串")
    if settings["aggregation"] != "normalized_mean":
        raise ValueError("yoloe.aggregation 当前只支持 normalized_mean")
    if type(settings["exclude_reference_overlap"]) is not bool:
        raise ValueError("yoloe.exclude_reference_overlap 必须是布尔值")
    for key in ("source_dir", "weights"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"yoloe.{key} 必须是路径字符串")
        settings[key] = str((base / settings[key]).expanduser().resolve())
    return settings


def source_fingerprint(root: Path) -> str | None:
    """Include actual Python/config bytes, so local upstream edits invalidate cache."""
    package = root / "ultralytics"
    if not package.is_dir():
        return None
    entries = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            entries.append([path.relative_to(root).as_posix(), image_sha256(path)])
    return digest({"files": entries})


def prepare_references(config: dict) -> None:
    settings = config["yoloe"]
    pool = ReferencePool(config, settings["selection_seed"])
    count = settings["positive_count"]
    if len(pool.positives) < count:
        raise ValueError(f"YOLOE 正样本不足: 要求 {count} 张，实际 {len(pool.positives)} 张")
    labels = {label["name"] for label in config["labels"]}
    examples = []
    positive_rows, _ = pool.select(count)
    for row in positive_rows:
        path = pool.path(row["image"])
        with Image.open(path) as image:
            width, height = image.size
        doc = pool.documents[row["sample_id"]]
        if doc.get("imageWidth") != width or doc.get("imageHeight") != height:
            raise ValueError(f"参考标注尺寸与原图不符: {row['image']}")
        boxes = []
        for shape in doc["shapes"]:
            if shape.get("label") not in labels:
                continue
            points = shape.get("points")
            if (shape.get("shape_type") not in {"rectangle", "polygon"} or not isinstance(points, list)
                    or len(points) < 2 or any(not isinstance(p, list) or len(p) != 2 for p in points)):
                raise ValueError(f"YOLOE 参考框必须是有效 rectangle/polygon: {row['annotation']}")
            if any(type(v) not in (int, float) or not math.isfinite(v) for p in points for v in p):
                raise ValueError(f"YOLOE 参考坐标必须是有限数字: {row['annotation']}")
            x1, x2 = min(p[0] for p in points), max(p[0] for p in points)
            y1, y2 = min(p[1] for p in points), max(p[1] for p in points)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"YOLOE 参考框越界或为空: {row['annotation']}")
            boxes.append([x1, y1, x2, y2])
        if not boxes:
            raise ValueError(f"YOLOE 正样本没有目标类标注框: {row['annotation']}")
        examples.append({"sample_id": row["sample_id"], "image_path": str(path),
                         "json_path": str(pool.path(row["annotation"])), "bboxes": boxes,
                         "image_sha256": image_sha256(path)})
    config.update(reference_examples=examples, reference_originals=examples,
                  reference_crops=[], reference_negatives=[], reference_boxed_images=[],
                  reference_pool_image_hashes=sorted(pool.original_hashes))
    config["reference_manifest_sha256"] = digest({"schema": "yoloe_visual.v1", "examples": examples})
    root = Path(settings["source_dir"])
    upstream = read_json(root / "UPSTREAM.json") if (root / "UPSTREAM.json").is_file() else {}
    weights = Path(settings["weights"])
    source_hash = source_fingerprint(root)
    config["local_model"] = {
        "settings": settings, "weights_sha256": image_sha256(weights) if weights.is_file() else None,
        "source_sha256": source_hash, "upstream_revision": upstream.get("revision"),
        "adapter_sha256": image_sha256(Path(__file__)),
        "runtime_ready": weights.is_file() and source_hash is not None,
    }


class YOLOERunner:
    def __init__(self, config: dict):
        self.config, self.settings = config, config["yoloe"]
        self.setup_seconds = 0.0
        started = time.perf_counter()
        source = Path(self.settings["source_dir"])
        weights = Path(self.settings["weights"])
        if not weights.is_file():
            raise ValueError(f"YOLOE 权重不存在: {weights}；请先运行 tool/setup_yoloe.ps1 -DownloadWeights")
        if not (source / "ultralytics/models/yolo/yoloe/predict_vp.py").is_file():
            raise ValueError(f"不是本项目使用的 THU-MIG YOLOE 源码: {source}")
        # The local environment may have another editable ultralytics installation.
        # Never silently mix its classes/checkpoint definitions with this checkout.
        loaded = sys.modules.get("ultralytics")
        if loaded and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise RuntimeError("当前进程已导入其它 ultralytics，请在新进程运行本项目 main.py")
        sys.path.insert(0, str(source))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs/runs/runtime/yoloe_settings"))
        try:
            import numpy as np
            import torch
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        except ImportError as exc:
            raise RuntimeError("YOLOE 依赖未就绪，请使用 Python 3.10/3.11 环境并按 docs/YOLOE_GUIDE.md 安装。"
                               f" 原因: {exc}") from exc
        self.np, self.torch = np, torch
        torch.set_num_threads(self.settings["torch_threads"])
        self.model = YOLOE(str(weights))
        self.model.model.eval()
        self.setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prototypes = []
        for example in config["reference_examples"]:
            data = Path(example["image_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != example["image_sha256"]:
                raise ValueError(f"参考原图在准备后发生变化: {example['image_path']}")
            self.model.predictor = None
            self.model.predict(self._image(data), prompts={
                "bboxes": np.asarray(example["bboxes"], dtype=np.float32),
                "cls": np.zeros(len(example["bboxes"]), dtype=np.int64)},
                predictor=YOLOEVPSegPredictor, return_vpe=True, **self._predict_args())
            embedding = self.model.predictor.vpe.detach()
            if embedding.ndim != 3 or tuple(embedding.shape[:2]) != (1, 1):
                raise ValueError(f"YOLOE 单类参考特征形状异常: {tuple(embedding.shape)}")
            prototypes.append(torch.nn.functional.normalize(embedding, dim=-1))
        prototype = torch.stack(prototypes).mean(dim=0)
        if not torch.isfinite(prototype).all() or prototype.norm().item() < 1e-8:
            raise ValueError("YOLOE 聚合参考特征为空或非有限值")
        prototype = torch.nn.functional.normalize(prototype, dim=-1)
        self.model.set_classes([config["labels"][0]["name"]], prototype)
        self.model.predictor = None
        self._sync()
        self.reference_encoding_seconds = time.perf_counter() - started

    def _image(self, data: bytes):
        with Image.open(io.BytesIO(data)) as source:
            return self.np.asarray(source.convert("RGB"))[:, :, ::-1].copy()

    def _predict_args(self):
        s = self.settings
        return dict(device=s["device"], imgsz=s["imgsz"], conf=s["confidence"], iou=s["nms_iou"],
                    max_det=s["max_detections"], half=False, rect=False, verbose=False,
                    save=False, save_txt=False, save_crop=False, show=False)

    def _sync(self):
        if self.torch.cuda.is_available() and next(self.model.model.parameters()).is_cuda:
            self.torch.cuda.synchronize()

    def predict(self, data: bytes) -> dict:
        image = self._image(data)
        height, width = image.shape[:2]
        self._sync()
        results = self.model.predict(image, **self._predict_args())
        self._sync()
        if len(results) != 1:
            raise ValueError(f"YOLOE 返回了 {len(results)} 个结果，应为 1 个")
        boxes = results[0].boxes
        objects = []
        if boxes is not None:
            for box, score, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                if cls != 0:
                    raise ValueError(f"YOLOE 单类结果包含意外类别 ID: {cls}")
                if any(not math.isfinite(v) for v in [*box, score]):
                    raise ValueError("YOLOE 输出非有限坐标或置信度")
                x1, y1, x2, y2 = box
                clipped = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
                           max(0.0, min(width, x2)), max(0.0, min(height, y2))]
                if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                    raise ValueError("YOLOE 输出退化框")
                objects.append({"label": self.config["labels"][0]["name"],
                                "bbox_pixels_xyxy": clipped, "confidence": score})
        payload = {"annotation_status": "annotated" if objects else "no_target",
                   "reason_code": "target_detected" if objects else "no_target",
                   "reason": "YOLOE 视觉参考检测", "target_count": len(objects), "objects": objects}
        return validate_pixel_prediction(payload, {self.config["labels"][0]["name"]}, width, height)

    async def close(self):
        self.model = None


from .base import LocalDetector, infer_local as infer_image


class YOLOEVisualDetector(LocalDetector):
    """Unified detector using a fixed visual class prototype."""
