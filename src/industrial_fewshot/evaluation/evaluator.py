"""Evaluate existing prediction records without importing or calling a model SDK."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from industrial_fewshot.postprocess.schema import validate_prediction

from industrial_fewshot.evaluation.detection_metrics import check_evaluation_dependencies, evaluate_detection, image_detection_counts
from industrial_fewshot.data.labelme import load_evaluation_truth, read_json, truth_summary, write_json, validate_pixel_prediction
from industrial_fewshot.paths import resolve_location
from industrial_fewshot.labels import all_label_names, target_label_names


def evaluate_records(items: list[dict], records: list[dict], labels: list[str], settings: dict) -> dict:
    predictions = {item["relative_path"]: [] for item in items}
    for record in records:
        if record["status"] == "success":
            predictions[record["relative_path"]].extend(record.get("objects", []))
    samples = [{**item, "predictions": predictions[item["relative_path"]]}
               for item in items if item["ground_truth"] is not None]
    failed = sum(record["status"] != "success" for record in records)
    metrics = evaluate_detection(samples, labels, settings)
    negative_samples = [sample for sample in samples if not sample["ground_truth"]]
    failed_paths = {r["relative_path"] for r in records if r["status"] != "success"}
    incorrect_images = []
    image_counts = {}
    for sample in samples:
        path = sample["relative_path"]
        if path in failed_paths:
            incorrect_images.append({"relative_path": path, "reason": "inference_failed"})
            continue
        counts = image_detection_counts(sample, labels, metrics["settings"])
        image_counts[path] = counts
        if counts["fp"] or counts["fn"]:
            incorrect_images.append({"relative_path": path, "reason": "detection_error", **counts})
    correct_images = len(samples) - len(incorrect_images)
    false_positives, true_negative_count, failed_negative_count = [], 0, 0
    for sample in negative_samples:
        if sample["relative_path"] in failed_paths:
            failed_negative_count += 1
            continue
        fp_count = image_counts[sample["relative_path"]]["fp"]
        if fp_count:
            false_positives.append({"relative_path": sample["relative_path"], "false_positive_box_count": fp_count})
        else:
            true_negative_count += 1
    successful_negatives = len(false_positives) + true_negative_count
    metrics.update(
        evaluation_status="incomplete" if failed else "complete",
        failed_request_count=failed,
        failure_policy="Failed requests contribute no predictions (GT remains); metrics are marked incomplete.",
        image_accuracy=correct_images / len(samples),
        correct_image_count=correct_images,
        incorrect_image_count=len(incorrect_images),
        failed_image_count=sum(sample["relative_path"] in failed_paths for sample in samples),
        incorrect_images=incorrect_images,
        image_accuracy_policy="All evaluated images form the denominator, including inference failures. Correct requires a successful request, class-aware one-to-one matches for every GT box and zero FP/FN at configured IoU/score/max-detection settings. An inference failure is always an incorrect image, even for empty GT.",
        **truth_summary(items),
        unlabeled_policy="Infer all images. Missing GT follows the configured policy: negative uses empty GT; skip excludes from metrics; error stops before inference.",
        negative_image_count=len(negative_samples),
        negative_false_positive_image_count=len(false_positives),
        negative_true_negative_image_count=true_negative_count,
        negative_failed_image_count=failed_negative_count,
        negative_false_positive_rate=len(false_positives) / successful_negatives if successful_negatives else None,
        negative_error_image_count=len(false_positives) + failed_negative_count,
        negative_error_rate=(len(false_positives) + failed_negative_count) / len(negative_samples) if negative_samples else None,
        negative_false_positive_box_count=sum(row["false_positive_box_count"] for row in false_positives),
        negative_false_positive_images=false_positives,
        negative_metric_policy="Image-level FP rate at score_threshold, capped per class; denominator excludes failed negative images. Failures are not true negatives and mark evaluation incomplete.",
        negative_error_policy="(Negative FP images + failed negative images) / all evaluated negative images. Every failed negative is an error and remains in the denominator.",
    )
    return metrics


def evaluate_saved_run(run_dir: Path, config_path: Path | None = None, dry_run: bool = False) -> dict:
    """Recover both historical runs with only run.json/records.jsonl and new runs."""
    run_dir = resolve_location(run_dir)
    report = read_json(run_dir / "run.json")
    records_path = run_dir / "records.jsonl"
    import json
    latest = {}
    for number, line in enumerate(records_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            relative = record["relative_path"]
            target_id = record["target_id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"推理记录第 {number} 行无效: {records_path}") from exc
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative:
            raise ValueError(f"推理记录包含无效图片路径: {relative!r}")
        if record.get("status") not in {"success", "failed"}:
            raise ValueError(f"推理记录状态无效: 第 {number} 行")
        latest[target_id, relative] = record
    targets = report["targets"]
    target_ids = {target["id"] for target in targets}
    paths = sorted({relative for _, relative in latest})
    expected = report["image_count"] * len(target_ids)
    if (
        len(paths) != report["image_count"] or len(latest) != expected
        or {target_id for target_id, _ in latest} != target_ids
        or any((target_id, relative) not in latest for target_id in target_ids for relative in paths)
    ):
        raise ValueError(
            f"该运行推理记录不完整：已有 {len(latest)} 条，预期 {expected} 条。"
            "请指定已完成推理的运行目录，避免把未处理图片遗漏出评估。"
        )
    saved_paths = report.get("image_paths")
    if saved_paths is not None and set(paths) != set(saved_paths):
        raise ValueError("推理记录与运行时保存的图片清单不一致")
    records = sorted(latest.values(), key=lambda record: (record["relative_path"], record["target_id"]))
    source = resolve_location(Path(report["image_dir"]))
    settings = report.get("evaluation", {})
    gt_root = resolve_location(Path(report.get("ground_truth_dir") or source))
    missing = report.get("missing_ground_truth", "skip")
    # Older runs did not snapshot these fields. Only evaluation/data settings are
    # read here: current model, credentials, prompts and references are irrelevant.
    if config_path is not None:
        config_path = config_path.expanduser().resolve()
        from industrial_fewshot.configuration import load_document
        raw, _ = load_document(config_path)
        settings = raw.get("evaluation", settings)
        dataset = raw.get("dataset", raw)
        if "ground_truth_dir" not in report and dataset.get("ground_truth_dir"):
            gt_root = (config_path.parent / Path(dataset["ground_truth_dir"]).expanduser()).resolve()
        missing = dataset.get("missing_ground_truth", missing)
    labels = set(all_label_names(targets))
    labels_by_target = {target['id']: set(target_label_names(target)) for target in targets}
    items = []
    for relative in paths:
        image = (source / relative).resolve()
        if not image.is_relative_to(source.resolve()):
            raise ValueError(f"推理图片路径超出原始数据目录: {relative}")
        with Image.open(image) as opened:
            width, height = opened.size
        for target_id in target_ids:
            record = latest[target_id, relative]
            if record["status"] == "success":
                backend = next(t.get("backend", "generic") for t in targets if t["id"] == target_id)
                validator = validate_pixel_prediction if backend in {"yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"} else validate_prediction
                validated = validator(record, labels_by_target[target_id], width, height)
                for original, obj in zip(record["objects"], validated["objects"]):
                    if "confidence" not in obj:
                        raise ValueError(f"已有记录缺少 confidence，无法计算排序 AP: {relative}")
                    if original.get("bbox_pixels_xyxy") != obj["bbox_pixels_xyxy"]:
                        raise ValueError(f"原图尺寸或保存的预测坐标已变化: {relative}")
        truth_path = gt_root / Path(relative).with_suffix(".json")
        truth, truth_source = load_evaluation_truth(truth_path, width, height, labels, missing)
        items.append({"relative_path": relative, "width": width, "height": height,
                      "ground_truth": truth, "ground_truth_source": truth_source})
    if all(item["ground_truth"] is None for item in items):
        raise ValueError("该运行没有可评估的真值图片")
    report.update(
        mode="train", run_dir=str(run_dir), evaluation_only=True,
        **truth_summary(items), missing_ground_truth=missing,
        failed_request_count=sum(record["status"] == "failed" for record in records),
        success_request_count=sum(record["status"] == "success" for record in records),
    )
    if dry_run:
        return {**report, "dry_run": True}
    check_evaluation_dependencies()
    metrics = evaluate_records(items, records, all_label_names(targets), settings)
    report.update(metrics=metrics, evaluation_status=metrics["evaluation_status"])
    # Keep source records and prediction JSON untouched. Only derived evaluation
    # artifacts are written, so repeating evaluation does not repeat inference.
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "summary.json", {**report, "annotations": records})
    from .reporting import save_text_report
    save_text_report(run_dir, report)
    from industrial_fewshot.prompt_records import update_prompt_index
    from industrial_fewshot.paths import PROJECT_ROOT
    default_root = PROJECT_ROOT / 'outputs/runs'
    index_root = report.get('prompt_index_root') or (default_root if run_dir.is_relative_to(default_root) else run_dir.parent)
    update_prompt_index(Path(index_root))
    return report
