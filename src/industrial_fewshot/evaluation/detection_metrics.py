"""Class-aware bbox AP and an explicit inference-only detection proxy loss.

AP follows COCO's IoU grid and 101 recall points for ordinary rectangles,
all areas, max 100 detections per image/category. No crowd/ignore/area splits.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Any


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def check_evaluation_dependencies() -> None:
    """Fail before inference if this Python environment cannot compute the loss."""
    try:
        import numpy  # noqa: F401
        from scipy.optimize import linear_sum_assignment  # noqa: F401
        from pycocotools.cocoeval import COCOeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境缺少或无法加载评估依赖 numpy/scipy/pycocotools。"
            "请在运行 main.py 的同一个环境执行: python -m pip install numpy scipy pycocotools。"
            "已有推理结果可通过 --evaluate-only 运行目录 补算，无需再次推理。"
        ) from exc


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _curve(samples: list[dict], label: str, threshold: float, max_dets: int, score_floor: float = 0.0) -> dict:
    truths = [[g for g in s["ground_truth"] if g["label"] == label] for s in samples]
    ranked = []
    for index, sample in enumerate(samples):
        predictions = sorted(
            (p for p in sample["predictions"] if p["label"] == label and p["confidence"] >= score_floor),
            key=lambda p: -p["confidence"],
        )[:max_dets]
        ranked.extend((p["confidence"], index, p) for p in predictions)
    ranked.sort(key=lambda entry: -entry[0])  # stable, including ties
    matched = [set() for _ in samples]
    tp = fp = 0
    recalls, precisions, overlaps = [], [], []
    gt_count = sum(map(len, truths))
    for _, index, prediction in ranked:
        candidates = [
            (box_iou(prediction["bbox_pixels_xyxy"], truth["bbox_pixels_xyxy"]), j)
            for j, truth in enumerate(truths[index]) if j not in matched[index]
        ]
        overlap, j = max(candidates, default=(0.0, -1))
        if j >= 0 and overlap >= threshold:
            tp += 1
            matched[index].add(j)
            overlaps.append(overlap)
        else:
            fp += 1
        recalls.append(_ratio(tp, gt_count))
        precisions.append(_ratio(tp, tp + fp))
    # Precision envelope, sampled at 101 recall thresholds, including recall=0.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = None
    if gt_count:
        values = []
        for step in range(101):
            i = bisect_left(recalls, step / 100)
            values.append(precisions[i] if i < len(precisions) else 0.0)
        ap = sum(values) / 101
    return {"ap": ap, "tp": tp, "fp": fp, "fn": gt_count - tp, "gt_count": gt_count, "matched_ious": overlaps}


def image_detection_counts(sample: dict, labels: list[str], settings: dict) -> dict:
    """Use the detection metric's matching rules for one image's correctness."""
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for label in labels:
        matched = _curve([sample], label, settings["iou_threshold"],
                         settings["max_detections_per_image"], settings["score_threshold"])
        for key in counts:
            counts[key] += matched[key]
    return counts


def _proxy_loss(samples: list[dict], labels: list[str], counts: dict, settings: dict) -> dict:
    # Hungarian assignment avoids greedy localization pairings. Unmatched/mislocalized
    # detection penalties are taken separately from thresholded TP/FP/FN matching.
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    l1_sum = iou_sum = pair_count = 0
    for sample in samples:
        for label in labels:
            gt = [g for g in sample["ground_truth"] if g["label"] == label]
            pred = sorted(
                (p for p in sample["predictions"] if p["label"] == label and p["confidence"] >= settings["score_threshold"]),
                key=lambda p: -p["confidence"],
            )[:settings["max_detections_per_image"]]
            if not gt or not pred:
                continue
            scale = [sample["width"], sample["height"], sample["width"], sample["height"]]
            l1 = np.array([[
                sum(abs(a - b) / s for a, b, s in zip(p["bbox_pixels_xyxy"], g["bbox_pixels_xyxy"], scale)) / 4
                for g in gt
            ] for p in pred])
            iou_loss = np.array([[1 - box_iou(p["bbox_pixels_xyxy"], g["bbox_pixels_xyxy"]) for g in gt] for p in pred])
            rows, cols = linear_sum_assignment(l1 + iou_loss)
            l1_sum += float(l1[rows, cols].sum())
            iou_sum += float(iou_loss[rows, cols].sum())
            pair_count += len(rows)
    components = {
        "box_l1": _ratio(l1_sum, pair_count),
        "box_iou": _ratio(iou_sum, pair_count),
        "false_positive": _ratio(counts["fp"], counts["tp"] + counts["fp"]),
        "false_negative": _ratio(counts["fn"], counts["tp"] + counts["fn"]),
    }
    weights = settings.get("loss_weights", {})
    return {
        "type": "detection_proxy_loss", "is_model_training_loss": False,
        "total": sum(components[k] * weights.get(k, 1.0) for k in components),
        "components": components, "matched_localization_pairs": pair_count,
        "weights": {k: weights.get(k, 1.0) for k in components},
        "definition": "Hungarian mean normalized xyxy L1 + mean (1-IoU) + FP/(TP+FP) + FN/(TP+FN); weighted sum. Localization pairing is class-aware and does not require an IoU threshold. No backpropagation.",
    }


def evaluation_settings(raw: dict | None = None) -> dict:
    settings = {"iou_threshold": 0.5, "score_threshold": 0.0, "max_detections_per_image": 100, **(raw or {})}
    extra_thresholds = settings.get("score_thresholds", [0.5, 0.7])
    if not isinstance(extra_thresholds, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not 0 <= value <= 1 for value in extra_thresholds
    ):
        raise ValueError("evaluation.score_thresholds 必须是 0~1 有限数字的数组")
    settings["score_thresholds"] = sorted(set(extra_thresholds))
    for key in ("iou_threshold", "score_threshold"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"evaluation.{key} 必须是 0~1 的有限数字")
    if settings["iou_threshold"] <= 0:
        raise ValueError("evaluation.iou_threshold 必须大于 0")
    limit = settings["max_detections_per_image"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_detections_per_image 必须是正整数")
    weights = settings.get("loss_weights", {})
    if not isinstance(weights, dict) or set(weights) - {"box_l1", "box_iou", "false_positive", "false_negative"}:
        raise ValueError("loss_weights 包含未知损失项")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("loss_weights 必须是有限非负数字")
    return settings


def output_score_thresholds(settings: dict) -> list[float]:
    return sorted({settings["score_threshold"], *settings.get("score_thresholds", [0.5, 0.7])})


def evaluate_score_thresholds(samples: list[dict], labels: list[str], settings: dict) -> dict:
    """Operating points on the same saved candidates; AP is reported separately."""
    from .coco_metrics import f2_score
    results = {}
    for threshold in output_score_thresholds(settings):
        categories = {}
        for label in labels:
            curve = _curve(samples, label, settings["iou_threshold"],
                           settings["max_detections_per_image"], threshold)
            tp, fp, fn = (curve[key] for key in ("tp", "fp", "fn"))
            precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
            categories[label] = {"tp": tp, "fp": fp, "fn": fn,
                                 "ground_truth_count": curve["gt_count"],
                                 "precision": precision, "recall": recall,
                                 "f2": f2_score(precision, recall)}
        totals = {key: sum(row[key] for row in categories.values()) for key in ("tp", "fp", "fn")}
        precision = _ratio(totals["tp"], totals["tp"] + totals["fp"])
        recall = _ratio(totals["tp"], totals["tp"] + totals["fn"])
        results[f"{threshold:g}"] = {"settings": {**settings, "score_threshold": threshold},
                                    **totals, "precision": precision, "recall": recall,
                                    "f2": f2_score(precision, recall), "per_class": categories}
    return results


def evaluate_detection(samples: list[dict], labels: list[str], raw_settings: dict | None = None) -> dict[str, Any]:
    if not samples:
        raise ValueError("没有可评估的真值图片")
    settings = evaluation_settings(raw_settings)
    thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    per_class = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    all_ious = []
    for label in labels:
        curves = [_curve(samples, label, t, settings["max_detections_per_image"]) for t in thresholds]
        operating = _curve(samples, label, settings["iou_threshold"], settings["max_detections_per_image"], settings["score_threshold"])
        precision = _ratio(operating["tp"], operating["tp"] + operating["fp"])
        recall = _ratio(operating["tp"], operating["tp"] + operating["fn"])
        per_class[label] = {
            "ground_truth_count": operating["gt_count"],
            **{k: operating[k] for k in totals},
            "precision": precision, "recall": recall, "f1": _ratio(2 * precision * recall, precision + recall),
            "AP50": curves[0]["ap"], "AP75": curves[5]["ap"],
            "AP50_95": sum(c["ap"] for c in curves) / 10 if curves[0]["ap"] is not None else None,
            "ap_by_iou": {str(t): c["ap"] for t, c in zip(thresholds, curves)},
        }
        for key in totals:
            totals[key] += operating[key]
        all_ious.extend(operating["matched_ious"])
    precision = _ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = _ratio(totals["tp"], totals["tp"] + totals["fn"])
    result = {
        "evaluated_image_count": len(samples), "settings": settings,
        "method": "Class-aware bbox, score-ranked greedy one-to-one matching, COCO-style 101-point AP, all areas, no crowd/ignore. AP uses all confidence scores; score_threshold affects only precision/recall/F1 and proxy loss. Classes without GT excluded from mAP.",
        "confidence_source": "model_self_reported_not_calibrated",
        **totals, "precision": precision, "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "mean_matched_iou": _ratio(sum(all_ious), len(all_ious)), "per_class": per_class,
    }
    for destination, source in (("mAP50", "AP50"), ("mAP75", "AP75"), ("mAP50_95", "AP50_95")):
        values = [c[source] for c in per_class.values() if c[source] is not None]
        result[destination] = sum(values) / len(values) if values else None
    result["loss"] = _proxy_loss(samples, labels, totals, settings)
    from .coco_metrics import evaluate_coco, f2_score
    result['f2'] = f2_score(precision, recall)
    for category in per_class.values():
        category['f2'] = f2_score(category['precision'], category['recall'])
    result['coco'] = evaluate_coco(samples, labels)
    result['score_threshold_results'] = evaluate_score_thresholds(samples, labels, settings)
    return result
