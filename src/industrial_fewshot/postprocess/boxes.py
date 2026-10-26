"""Pixel geometry and deterministic class-aware non-maximum suppression."""
import math

from industrial_fewshot.evaluation.detection_metrics import box_iou


def clip_box(box, width: int, height: int) -> list[float]:
    if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
        raise ValueError("检测框必须包含四个有限坐标")
    clipped = [max(0.0, min(float(v), bound)) for v, bound in zip(box, (width, height, width, height))]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("检测框越界后退化")
    return clipped


def nms(objects: list[dict], iou_threshold: float = 0.7, max_detections: int = 100) -> list[dict]:
    kept = []
    for index, candidate in sorted(enumerate(objects), key=lambda pair: (-pair[1]["confidence"], pair[0])):
        if any(previous["label"] == candidate["label"] and
               box_iou(previous["bbox_pixels_xyxy"], candidate["bbox_pixels_xyxy"]) > iou_threshold for previous in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept
