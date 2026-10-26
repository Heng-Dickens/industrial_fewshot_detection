"""Re-evaluate saved YOLOE/VLM records on one verified cohort; never run a model."""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from industrial_fewshot.evaluation.detection_metrics import evaluation_settings
from industrial_fewshot.evaluation.evaluator import evaluate_records
from industrial_fewshot.experiments import comparison_row
from industrial_fewshot.data.labelme import load_evaluation_truth, read_json, validate_pixel_prediction, write_json
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.postprocess.schema import validate_prediction
from industrial_fewshot.paths import resolve_location


def saved_groups(path: Path) -> list[dict]:
    path = resolve_location(path)
    if path.is_dir() and (path / "comparison.json").is_file():
        path = path / "comparison.json"
    if path.is_file() and path.name == "comparison.json":
        comparison = read_json(path)
        if not comparison.get("groups"):
            raise ValueError(f"没有已完成的实验组: {path}")
        return [load_saved_group(Path(group["run_dir"])) for group in comparison["groups"]]
    return [load_saved_group(path if path.is_dir() else path.parent)]


def load_saved_group(directory: Path) -> dict:
    directory = resolve_location(directory)
    metadata = read_json(directory / "run.json")
    records = {}
    for line in (directory / "records.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            relative = record["relative_path"]
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"结果中包含不安全的图片路径: {relative}")
            records[record["target_id"], relative] = record
    target_ids = {t["id"] for t in metadata["targets"]}
    paths = set(metadata.get("image_paths", [path for _, path in records]))
    expected = {(target, path) for target in target_ids for path in paths}
    if len(paths) != metadata["image_count"] or set(records) != expected:
        raise ValueError(f"推理记录不完整，不能遗漏未处理图片: {directory}")
    if any(record.get("status") not in {"success", "failed"} for record in records.values()):
        raise ValueError(f"无效推理状态: {directory}")
    metadata.update(run_dir=str(directory), saved_records=list(records.values()))
    return metadata


def compare(vlm_paths: list[Path], yoloe_path: Path, output: Path, *, intersection=False,
            evaluation_config: Path | None = None) -> dict:
    if any(any(t["backend"] not in {"yoloe", "yoloe_text", "yoloe_fusion"} for t in g["targets"]) for g in saved_groups(yoloe_path)):
        raise ValueError("--yoloe-run 必须指向 YOLOE 结果")
    return compare_runs([yoloe_path, *vlm_paths], output, intersection=intersection, evaluation_config=evaluation_config)


def compare_runs(run_paths: list[Path], output: Path, *, intersection=False,
                 evaluation_config: Path | None = None) -> dict:
    if len(run_paths) < 2:
        raise ValueError("Provide at least two saved runs to compare")
    groups = [group for path in run_paths for group in saved_groups(path)]
    yoloe_ids = next((g["targets"][0].get("reference_sample_ids", []) for g in groups
                      if g["targets"][0]["backend"] == "yoloe"), [])
    first = groups[0]
    source = resolve_location(Path(first["image_dir"]))
    truth_root = resolve_location(Path(first["ground_truth_dir"]))
    labels = {t["label"] for t in first["targets"]}
    for group in groups:
        if (resolve_location(Path(group["image_dir"])) != source or resolve_location(Path(group["ground_truth_dir"])) != truth_root
                or {t["label"] for t in group["targets"]} != labels):
            raise ValueError("待测图片目录、真值目录和类别必须一致")
    cohorts = [{r["relative_path"] for r in group["saved_records"]} for group in groups]
    if not intersection and any(paths != cohorts[0] for paths in cohorts):
        raise ValueError("已有结果的待测图片集合不一致。确认后可用 --intersection 在共同图片上重算，并记录排除清单。")
    paths = sorted(set.intersection(*cohorts))
    if not paths:
        raise ValueError("没有共同的待测图片")
    if any(not p or Path(p) == Path(".") for p in paths):
        raise ValueError("待测图片相对路径不能为空")
    settings = first["evaluation"]
    missing = first["missing_ground_truth"]
    if evaluation_config is not None:
        from industrial_fewshot.configuration import load_document
        raw, _ = load_document(evaluation_config)
        settings = evaluation_settings(raw.get("evaluation"))
        missing = raw.get("dataset", raw).get("missing_ground_truth", missing)
    settings = evaluation_settings(settings)
    items = []
    for relative in paths:
        image_path = (source / relative).resolve()
        if not image_path.is_relative_to(source):
            raise ValueError(f"图片路径越界: {relative}")
        with Image.open(image_path) as image:
            width, height = image.size
        truth, provenance = load_evaluation_truth(truth_root / Path(relative).with_suffix(".json"), width, height, labels, missing)
        items.append({"path": image_path, "relative_path": relative, "width": width, "height": height,
                      "image_sha256": image_sha256(image_path), "ground_truth": truth, "ground_truth_source": provenance})
    if all(i["ground_truth"] is None for i in items):
        raise ValueError("共同图片没有可评估的真值")
    item_map = {i["relative_path"]: i for i in items}
    rows, details = [], []
    for index, group in enumerate(groups):
        targets = {t["id"]: t for t in group["targets"]}
        selected = []
        for record in group["saved_records"]:
            if record["relative_path"] not in item_map:
                continue
            record = dict(record)
            item = item_map[record["relative_path"]]
            # Failures are retained and evaluated as errors, too. Missing hashes
            # cannot establish that an old result used today's image bytes.
            if record.get("image_sha256") != item["image_sha256"]:
                raise ValueError(f"已有结果缺少图片指纹或图片已改变: {group['run_dir']} / {item['relative_path']}")
            if record["status"] == "success":
                target = targets[record["target_id"]]
                validator = validate_pixel_prediction if target["backend"] in {"yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"} else validate_prediction
                annotation = validator(record, {target["label"]}, item["width"], item["height"])
                if any("confidence" not in obj for obj in annotation["objects"]):
                    raise ValueError("检测结果缺少 confidence，无法比较排序 AP")
                record.update(annotation)
            selected.append(record)
        metrics = evaluate_records(items, selected, sorted(labels), settings)
        working = {**group, "metrics": metrics, "image_count": len(items),
                   "reused_prediction_count": len(selected), "executed_inference_count": 0,
                   "failed_request_count": sum(r["status"] != "success" for r in selected),
                   "evaluation_status": metrics["evaluation_status"]}
        mode = group.get("experiment", {}).get("id") or group["targets"][0]["reference_mode"]
        row = comparison_row(mode, working)
        meta = group.get("experiment", {})
        ids = group["targets"][0].get("reference_sample_ids", [])
        if meta.get("reference_sets"):
            ids = meta["reference_sets"][0]["positive_sample_ids"]
        if group["targets"][0]["backend"] in {"yoloe", "pet_dino"}:
            row.update(positive_original_count=len(ids), source_reference_count=len(ids), reference_image_count=len(ids))
        elif group['targets'][0]['backend'] == 'yoloe_fusion':
            row.update(positive_original_count=len(ids), source_reference_count=len(ids), json_reference_count=0,
                       reference_image_count=group['targets'][0]['reference_count']['reference_crops'])
        recorded_times = [r["inference_seconds"] for r in selected
                          if isinstance(r.get("inference_seconds"), (int, float))]
        backend = group["targets"][0]["backend"]
        row.update(method={"yoloe": "YOLOE", "yoloe_text": "YOLOE（文字提示）", "yoloe_fusion": "YOLOE（裁剪+文字融合）", "groundingdino": "GroundingDINO", 'yolo_onnx': 'YOLO ONNX（监督检测）', 'pet_dino': 'PET-DINO', 'wedetect': 'WeDetect'}.get(backend, "VLM"),
                   backend=backend, selection_seed=meta.get("selection_seed", group["targets"][0].get("local_model", {}).get("settings", {}).get("selection_seed")),
                   model=group["targets"][0]["model"], reference_set_matches_yoloe=bool(ids) and ids == yoloe_ids,
                   timed_record_count=len(recorded_times),
                   mean_recorded_image_seconds=statistics.mean(recorded_times) if recorded_times else None,
                   excluded_saved_image_count=len(cohorts[index] - set(paths)))
        rows.append(row)
        details.append({"run_dir": group["run_dir"], "metrics": metrics,
                        "excluded_saved_images": sorted(cohorts[index] - set(paths))})
    report = {"image_dir": str(source), "ground_truth_dir": str(truth_root), "image_paths": paths,
              "image_count": len(paths), "evaluation": settings, "missing_ground_truth": missing,
              "intersection": intersection, "rows": rows, "groups": details,
              "dataset_sha256": digest({"images": [{k: item[k] for k in ("relative_path", "image_sha256", "ground_truth")} for item in items]}),
              "policy": "All rows re-evaluated on the same verified image bytes, current GT and thresholds. "
                        "Failures stay in accuracy/error denominators. No model calls. Timing uses available historical records, not a controlled hardware benchmark."}
    output = output.expanduser().resolve()
    for group in groups:
        protected = Path(group["run_dir"]).resolve()
        if output == protected or output.is_relative_to(protected):
            raise ValueError("比较输出目录必须位于已有运行目录之外")
    for protected in (source, truth_root):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("比较输出目录必须位于原图/真值目录之外")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "detector_comparison.json", report)
    from .reporting import save_text_report
    save_text_report(output, {**report, 'output_dir': str(output)}, 'detector_comparison.txt')
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output / "detector_comparison.csv").write_text(stream.getvalue(), encoding="utf-8-sig")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", help="Any saved detector run; repeat for each method")
    parser.add_argument("--vlm-run", type=Path, action="append", help="VLM 单次运行目录或完整 ablation 目录；可重复")
    parser.add_argument("--yoloe-run", type=Path, help="YOLOE 单次运行目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, help="统一评估阈值与缺失真值策略；默认采用 YOLOE 运行设置")
    parser.add_argument("--intersection", action="store_true", help="显式在各组共同图片上重算，记录排除清单")
    parser.add_argument('--json', action='store_true', help='输出 JSON 摘要；默认输出检测评估表格')
    args = parser.parse_args(argv)
    try:
        paths = [*(args.run or []), *([args.yoloe_run] if args.yoloe_run else []), *(args.vlm_run or [])]
        report = compare_runs(paths, args.output_dir, intersection=args.intersection,
                         evaluation_config=args.evaluation_config)
        if args.json:
            print(json.dumps({'image_count': report['image_count'], 'method_count': len(report['rows']),
                              'output_dir': str(args.output_dir)}, ensure_ascii=False))
        else:
            from .reporting import format_report
            print(format_report({**report, 'output_dir': str(args.output_dir)}))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
