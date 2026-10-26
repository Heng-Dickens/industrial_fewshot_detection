"""Local text-only GroundingDINO baseline using the Transformers public API."""
from __future__ import annotations

import importlib.metadata
import io
import math
from pathlib import Path
import time

from PIL import Image

from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.supports import ReferencePool
from industrial_fewshot.postprocess.boxes import clip_box, nms
from .base import LocalDetector


def groundingdino_settings(raw, base: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("groundingdino 必须是配置对象")
    settings = {"checkpoint": "../weights/grounding-dino-tiny", "device": "cpu",
                "text_prompt": "hanging industrial mold. suspended object.", "box_threshold": 0.001,
                "text_threshold": 0.25, "nms_iou": 0.7, "max_detections": 100,
                "torch_threads": 4, "exclude_reference_overlap": True, **raw}
    allowed = {"checkpoint", "device", "text_prompt", "box_threshold", "text_threshold", "nms_iou",
               "max_detections", "torch_threads", "exclude_reference_overlap"}
    if set(settings) - allowed:
        raise ValueError(f"未知 groundingdino 配置字段: {sorted(set(settings) - allowed)}")
    for key in ("box_threshold", "text_threshold", "nms_iou"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"groundingdino.{key} 必须是 0~1 的有限数字")
    for key in ("max_detections", "torch_threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"groundingdino.{key} 必须是正整数")
    for key in ("checkpoint", "device", "text_prompt"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"groundingdino.{key} 必须是非空字符串")
    if type(settings["exclude_reference_overlap"]) is not bool:
        raise ValueError("groundingdino.exclude_reference_overlap 必须是布尔值")
    settings["checkpoint"] = str((base / settings["checkpoint"]).expanduser().resolve())
    return settings


def prepare_references(config: dict):
    settings = config["groundingdino"]
    hashes = []
    if config.get("reference_image_dir"):
        hashes = sorted(ReferencePool(config, 0).original_hashes)
    config.update(reference_examples=[], reference_originals=[], reference_crops=[], reference_boxed_images=[],
                  reference_negatives=[], reference_pool_image_hashes=hashes)
    config["reference_manifest_sha256"] = digest({"schema": "groundingdino_text.v1", "text_prompt": settings["text_prompt"]})
    root = Path(settings["checkpoint"])
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}) if root.is_dir() else []
    ready = any(p.suffix in {".safetensors", ".bin"} for p in files) and (root / "config.json").is_file()
    versions = {}
    for package in ("torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config["local_model"] = {"settings": settings, "runtime_ready": ready, "library_versions": versions,
        "checkpoint_sha256": digest({"files": [[p.relative_to(root).as_posix(), image_sha256(p)] for p in files]}) if files else None,
        "adapter_sha256": image_sha256(Path(__file__))}


class GroundingDINORunner:
    def __init__(self, config: dict):
        self.config, self.settings = config, config["groundingdino"]
        if not config["local_model"]["runtime_ready"]:
            raise ValueError(f"GroundingDINO 本地权重未就绪: {self.settings['checkpoint']}；按 README 准备模型后重试")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("GroundingDINO 需要项目 ovd 可选依赖，请 pip install -e '.[ovd]'") from exc
        self.torch = torch
        torch.set_num_threads(self.settings["torch_threads"])
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.settings["checkpoint"], local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.settings["checkpoint"], local_files_only=True).to(self.settings["device"]).eval()
        self.setup_seconds = time.perf_counter() - started
        self.reference_encoding_seconds = 0.0

    def predict(self, data: bytes) -> dict:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
        width, height = image.size
        inputs = self.processor(images=image, text=self.settings["text_prompt"], return_tensors="pt").to(self.settings["device"])
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=self.settings["box_threshold"],
            text_threshold=self.settings["text_threshold"], target_sizes=[(height, width)])
        if len(results) != 1:
            raise ValueError("GroundingDINO 返回结果数量异常")
        objects = [{"label": self.config["labels"][0]["name"],
                    "bbox_pixels_xyxy": clip_box(box, width, height), "confidence": float(score)}
                   for box, score in zip(results[0]["boxes"].cpu().tolist(), results[0]["scores"].cpu().tolist())]
        objects = nms(objects, self.settings["nms_iou"], self.settings["max_detections"])
        return {"annotation_status": "annotated" if objects else "no_target",
                "reason_code": "target_detected" if objects else "no_target", "reason": "GroundingDINO 文本检测",
                "target_count": len(objects), "objects": objects}

    async def close(self):
        self.model = None


class GroundingDINOTextDetector(LocalDetector):
    """Unified image-only input / pixel-box output interface for text OVD."""
