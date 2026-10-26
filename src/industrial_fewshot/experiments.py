"""Controlled reference-input comparisons using the common annotation pipeline."""
from __future__ import annotations

import asyncio
import copy
import csv
import io
import logging
from pathlib import Path

from industrial_fewshot.evaluation.detection_metrics import check_evaluation_dependencies
from industrial_fewshot.data.labelme import truth_summary, write_json
from industrial_fewshot.runner import inspect_dataset, run, validate_reference_separation
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.prompts.references import REFERENCE_MODES, configure_reference_inputs, prepare_reference_examples
from industrial_fewshot.data.supports import ReferencePool, reference_pool, group_specs
from industrial_fewshot.outputs import model_shot_name, group_name, MODE_NAMES, result_directory

LOGGER = logging.getLogger(__name__)
METRIC_COLUMNS = ("precision", "recall", "f1", "mAP50", "mAP75", "mAP50_95", "mean_matched_iou")


def comparison_row(mode: str, report: dict) -> dict:
    metrics = report["metrics"]
    experiment = report.get("experiment", {})
    return {
        "experiment_id": experiment.get("id", mode),
        'experiment_name': experiment.get('name', experiment.get('id', mode)),
        "reference_mode": experiment.get("variant", mode),
        "positive_original_count": experiment.get("positive_count", 0),
        "negative_original_count": experiment.get("negative_count", 0),
        "source_reference_count": experiment.get("source_reference_count", 0),
        "reference_image_count": experiment.get("reference_image_count", 0),
        "original_reference_count": sum(t["reference_count"].get("reference_originals", 0) for t in report["targets"]),
        "boxed_reference_count": sum(t["reference_count"].get("reference_boxed_images", 0) for t in report["targets"]),
        "negative_reference_count": sum(t["reference_count"].get("reference_negatives", 0) for t in report["targets"]),
        "json_reference_count": sum(t["reference_count"]["reference_examples"] for t in report["targets"]),
        "crop_reference_count": sum(t["reference_count"]["reference_crops"] for t in report["targets"]),
        "image_count": report["image_count"],
        "evaluated_image_count": metrics["evaluated_image_count"],
        "unlabeled_image_count": metrics["unlabeled_image_count"],
        **{key: metrics[key] for key in ("image_accuracy", "correct_image_count", "incorrect_image_count", "failed_image_count")},
        **{key: metrics[key] for key in ("ground_truth_json_image_count", "assumed_negative_image_count",
            "negative_image_count", "negative_false_positive_image_count", "negative_true_negative_image_count",
            "negative_failed_image_count", "negative_false_positive_box_count", "negative_false_positive_rate",
            "negative_error_image_count", "negative_error_rate")},
        **{key: metrics[key] for key in METRIC_COLUMNS},
        "proxy_loss": metrics["loss"]["total"],
        **{key: metrics[key] for key in ("tp", "fp", "fn")},
        "reused_prediction_count": report["reused_prediction_count"],
        "executed_inference_count": report["executed_inference_count"],
        "failed_request_count": report["failed_request_count"],
        "evaluation_status": report["evaluation_status"],
        "run_dir": report["run_dir"],
    }


def _save_comparison(directory: Path, report: dict) -> None:
    rows = report["rows"]
    baseline = next((row for row in rows if row["reference_mode"] == "text_only"), None)
    for row in rows:
        for key in (*METRIC_COLUMNS, "proxy_loss", "image_accuracy", "negative_false_positive_rate", "negative_error_rate"):
            row["delta_" + key] = (row[key] - baseline[key]
                                   if baseline is not None and row[key] is not None and baseline[key] is not None else None)
    write_json(directory / "comparison.json", report)
    from .evaluation.reporting import save_text_report
    save_text_report(directory, report, 'comparison.txt')
    if rows:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = directory / "comparison.csv.tmp"
        temporary.write_text(stream.getvalue(), encoding="utf-8-sig")
        temporary.replace(directory / "comparison.csv")


async def run_ablation(config: dict, dry_run: bool = False, limit: int | None = None,
                       client_factory=None, force: bool = False) -> dict:
    if config["mode"] != "train":
        raise ValueError("数值对比实验需要 train 模式及真值 JSON；请设置 mode=train 或 --mode train")
    check_evaluation_dependencies()
    modes = config.get("ablation", {}).get("modes", list(REFERENCE_MODES))
    items = inspect_dataset(config, limit)
    for item in items:
        item["image_sha256"] = await asyncio.to_thread(image_sha256, item["path"])
        if item.get("expected_image_sha256") and item["image_sha256"] != item["expected_image_sha256"]:
            raise ValueError(f"图片内容与固定划分不一致: {item['path']}")
    variants = []
    suite = config.get("experiment_suite", {})
    excluded = []
    if suite.get("enabled"):
        pools = [reference_pool(t["config"], suite["selection_seed"]) for t in config["targets"]]
        reference_hashes = set().union(*(p.original_hashes for p in pools))
        excluded = [item["relative_path"] for item in items if item["image_sha256"] in reference_hashes]
        if excluded and not suite["exclude_reference_overlap"]:
            raise ValueError(f"参考原图与待测图片内容重复，不能泄漏待测真值: {excluded}")
        items = [item for item in items if item["image_sha256"] not in reference_hashes]
        if not items or all(item["ground_truth"] is None for item in items):
            raise ValueError("排除参考重叠图片后没有可评估的待测图片")
        if excluded:
            LOGGER.info("所有实验组统一排除与参考集内容重复的待测图片: %s", ", ".join(excluded))
        from .result_layout import apply_result_layout
        config = apply_result_layout(config, items, create=not dry_run)
        directory = result_directory(config['output_dir'], model_shot_name(config))
        for spec in group_specs(suite):
            variant = copy.deepcopy(config)
            variant["output_dir"] = directory / group_name(spec, include_shot=len(suite['positive_counts']) > 1)
            variant["prediction_cache_dir"] = config.get('prediction_cache_dir', config["output_dir"])
            metadata = {**spec, "selection_seed": suite["selection_seed"], "source_reference_count": 0, "reference_image_count": 0,
                        "reference_sets": []}
            metadata['name'] = variant['output_dir'].name
            for target, pool in zip(variant["targets"], pools):
                if spec["positive_count"]:
                    selection = pool.prepare(target["config"], spec)
                    metadata["reference_sets"].append({"target_id": target["id"], **selection})
                    for key in ("source_reference_count", "reference_image_count"):
                        metadata[key] += selection[key]
                else:
                    configure_reference_inputs(target["config"], "text_only")
                    prepare_reference_examples(target["config"])
            variant["experiment"] = metadata
            validate_reference_separation(variant, items)
            variants.append((spec["id"], variant))
        modes = [name for name, _ in variants]
    else:
        from .result_layout import apply_result_layout
        config = apply_result_layout(config, items, create=not dry_run)
        directory = result_directory(config['output_dir'], model_shot_name(config))
    # Prepare every group before making any model call. Freeze the selected
    # reference bytes and use the same image/GT snapshot for every group.
    for mode in ([] if suite.get("enabled") else modes):
        variant = copy.deepcopy(config)
        variant["output_dir"] = directory / MODE_NAMES[mode]
        variant["prediction_cache_dir"] = config.get('prediction_cache_dir', config["output_dir"])
        variant['experiment'] = {'name': MODE_NAMES[mode]}
        for target in variant["targets"]:
            configure_reference_inputs(target["config"], mode)
            prepare_reference_examples(target["config"])
        validate_reference_separation(variant, items)
        variants.append((mode, variant))
    report = {
        "scene_name": config["scene_name"], "config_file": config["config_file"],
        "mode": "train", "ablation": True, "dry_run": dry_run,
        "output_dir": str(config["output_dir"]), "comparison_dir": str(directory),
        "result_layout": config.get('result_layout'),
        "image_dir": str(config["image_dir"]), "ground_truth_dir": str(config["ground_truth_dir"]),
        "image_count": len(items), **truth_summary(items),
        "missing_ground_truth": config["missing_ground_truth"],
        "image_paths": [i["relative_path"] for i in items], "evaluation": config["evaluation"],
        "dataset_sha256": digest({"images": [{"relative_path": i["relative_path"],
            "image_sha256": i["image_sha256"], "ground_truth": i["ground_truth"]} for i in items]}),
        "modes": modes, "status": "planned" if dry_run else "running", "groups": [], "rows": [],
        "group_count": len(variants), "excluded_reference_overlap": excluded,
        "experiment_suite": suite if suite.get("enabled") else None,
        "failed_request_count": 0,
        "reference_policy": ("Paired original/crop/boxed/JSON positive references; equal-count negative originals when enabled. "
                             "All groups use the same test cohort, disjoint from the full reference pool by image SHA256. "
                             "Test-image GT is used only for local evaluation." if suite.get("enabled") else
                             "Reference JSON text and/or positive crops only; no reference full images or negatives. Test-image GT is used only for local evaluation."),
    }
    resume = config.get('resume_comparison_dir')
    resume_results = {}
    interrupted_groups = []
    if resume:
        if force:
            raise ValueError('恢复已完成分组与 force 不能同时使用')
        from .data.labelme import read_json
        previous = read_json(Path(resume) / 'comparison.json')
        if previous['dataset_sha256'] != report['dataset_sha256'] or previous.get('experiment_suite') != report['experiment_suite']:
            raise ValueError('恢复目录的数据集或实验组合与当前配置不一致')
        directory = Path(resume)
        report['comparison_dir'] = str(directory)
        for _, variant in variants:
            variant['output_dir'] = directory / variant['output_dir'].name
        # Validate every existing group before changing any comparison file.
        for mode, variant in variants:
            existing = variant['output_dir'] / 'summary.json'
            if not variant['output_dir'].exists():
                continue
            saved = read_json(existing if existing.is_file() else variant['output_dir'] / 'run.json')
            planned = await run(variant, True, shared_items=copy.deepcopy(items), references_prepared=True,
                                verify_shared_images=False, result_dir=variant['output_dir'])
            if (saved['dataset_sha256'] != planned['dataset_sha256']
                    or saved.get('evaluation') != planned['evaluation']
                    or [t.get('request_signature') for t in saved['targets']] != [t['request_signature'] for t in planned['targets']]):
                raise ValueError('恢复目录的提示词、参考输入、评估设置或模型参数与本次不一致')
            if saved.get('failed_request_count') or saved.get('evaluation_status') != 'complete':
                interrupted_groups.append(variant['output_dir'])
                continue
            resume_results[mode] = planned if dry_run else saved
    if not dry_run:
        if interrupted_groups:
            from .paths import PROJECT_ROOT
            index_root = Path(config.get('prompt_index_root', config['output_dir']))
            archive_root = (PROJECT_ROOT / 'outputs/archive/interrupted_runs'
                            if index_root.is_relative_to(PROJECT_ROOT / 'outputs/runs') else index_root / '.interrupted')
            for source in interrupted_groups:
                if not source.resolve().is_relative_to(Path(resume).resolve()):
                    raise ValueError('未完成分组不在指定恢复目录内')
                parent = archive_root / config.get('result_layout', {}).get('dataset_name', config['scene_name']) / directory.name
                parent.mkdir(parents=True, exist_ok=True)
                archived = result_directory(parent, source.name)
                if not archived.resolve().is_relative_to(archive_root.resolve()):
                    raise ValueError('归档路径超出指定目录')
                source.rename(archived)
                LOGGER.info('保留中断分组的原始文件: %s；成功预测从缓存恢复。', archived)
        reserved = directory if resume else result_directory(config['output_dir'], model_shot_name(config), create=True)
        if reserved != directory:
            for _, variant in variants:
                variant['output_dir'] = reserved / variant['output_dir'].name
            directory = reserved
            report['comparison_dir'] = str(directory)
        _save_comparison(directory, report)
    for mode, variant in variants:
        LOGGER.info("参考输入对比实验: %s", variant['output_dir'].name)
        try:
            if mode in resume_results:
                result = resume_results[mode]
                LOGGER.info('保留已完成分组: %s', variant['output_dir'])
            else:
                result = await run(variant, dry_run, client_factory=client_factory, force=force,
                               shared_items=copy.deepcopy(items), references_prepared=True,
                               verify_shared_images=not dry_run, result_dir=variant['output_dir'])
        except (asyncio.CancelledError, KeyboardInterrupt):
            if not dry_run:
                report.update(status='interrupted', interrupted_mode=mode)
                _save_comparison(directory, report)
            raise
        except Exception as exc:
            if not dry_run:
                report.update(status="failed", failed_mode=mode, error=str(exc))
                if getattr(exc, 'status_code', None) is not None:
                    report['service_status_code'] = exc.status_code
                _save_comparison(directory, report)
            raise RuntimeError(f"对比实验 {variant['output_dir'].name} 失败: {exc}。汇总目录: {directory}；修复后重运行原命令自动复用成功推理。") from exc
        report["groups"].append(result)
        if not dry_run:
            report["rows"].append(comparison_row(mode, result))
            report["failed_request_count"] += result["failed_request_count"]
            _save_comparison(directory, report)
    report["reused_prediction_count"] = sum(r["reused_prediction_count"] for r in report["groups"])
    report["scheduled_inference_count"] = sum(r["scheduled_inference_count"] for r in report["groups"])
    if not dry_run:
        report["executed_inference_count"] = sum(r["executed_inference_count"] for r in report["groups"])
        report["status"] = "incomplete" if report["failed_request_count"] else "complete"
        _save_comparison(directory, report)
        LOGGER.info("对比指标汇总: %s", directory / "comparison.csv")
    return report
