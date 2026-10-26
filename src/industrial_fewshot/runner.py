"""Scenario orchestration: scan, infer, export, and optionally evaluate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from PIL import Image

import industrial_fewshot.detectors.mllm as inference
import industrial_fewshot.prompts.references as references
from industrial_fewshot.config import DEFAULTS, load_config, render_prompt
from industrial_fewshot.data.labelme import build_labelme_document, load_evaluation_truth, read_json, truth_summary, write_json
from industrial_fewshot.evaluation.detection_metrics import check_evaluation_dependencies
from industrial_fewshot.evaluation.evaluator import evaluate_records, evaluate_saved_run
from industrial_fewshot.cache import PredictionCache, digest, image_sha256
from industrial_fewshot.detectors.registry import is_local, local_runner_class
from industrial_fewshot.outputs import model_shot_name, result_directory
from industrial_fewshot.prompt_records import capture_prompt, save_prompt, update_prompt_index

LOGGER = logging.getLogger(__name__)
from .labels import all_label_names, target_label_names


def validate_reference_separation(config: dict, items: list[dict]) -> None:
    """Reject reference files that identify an image currently being evaluated."""
    image_paths = {item["path"].resolve() for item in items}
    truth_paths = {(config["ground_truth_dir"] / Path(item["relative_path"]).with_suffix(".json")).resolve()
                   for item in items}
    for target in config["targets"]:
        for example in target["config"].get("reference_examples", []):
            json_path = Path(example["json_path"]).resolve()
            image_path = example.get("image_path")
            if not image_path:
                raw_image_path = json.loads(example["compact_json"]).get("imagePath")
                image_path = json_path.parent / raw_image_path if raw_image_path else None
            if json_path in truth_paths or image_path and Path(image_path).resolve() in image_paths:
                raise ValueError(f"参考标注对应待测图片，不能传给模型；请使用独立参考样例: {json_path}")





async def infer_image(client, target: dict, item: dict) -> dict:
    from .detectors.registry import create_detector
    from .detectors.base import ImageInput
    return await create_detector(client, target).predict(ImageInput.from_item(item))



async def run(config: dict, dry_run: bool = False, limit: int | None = None, client_factory=None,
              force: bool = False, *, shared_items: list[dict] | None = None,
              references_prepared: bool = False, verify_shared_images: bool = True,
              local_runner_factory=None, result_dir: Path | None = None) -> dict:
    if not verify_shared_images and (not dry_run or shared_items is None):
        raise ValueError("只有只读规划可以复用已校验的图片指纹")
    items = shared_items if shared_items is not None else inspect_dataset(config, limit)
    if config["mode"] == "train":
        check_evaluation_dependencies()
    if not references_prepared:
        for target in config["targets"]:
            references.prepare_reference_examples(target["config"])
    LOGGER.info("正在检查图片内容及可复用的成功推理记录，共 %d 张", len(items))
    for index, item in enumerate(items, 1):
        if not verify_shared_images:
            continue
        checksum = await asyncio.to_thread(image_sha256, item["path"])
        if item.get("expected_image_sha256") and checksum != item["expected_image_sha256"]:
            raise ValueError(f"图片内容与固定划分不一致，请重新准备数据: {item['path']}")
        if shared_items is not None and checksum != item["image_sha256"]:
            raise ValueError(f"对比实验期间待测图片发生变化，请重新运行: {item['path']}")
        item["image_sha256"] = checksum
        if index % 25 == 0:
            LOGGER.info("已检查图片内容 %d/%d", index, len(items))
    excluded_overlap = []
    local_targets = [t for t in config["targets"] if is_local(t["backend"])]
    if local_targets:
        hashes = {h for t in local_targets for h in t["config"]["reference_pool_image_hashes"]}
        excluded_overlap = [i["relative_path"] for i in items if i["image_sha256"] in hashes]
        if excluded_overlap and any(not t["config"][t["backend"]]["exclude_reference_overlap"] for t in local_targets):
            raise ValueError(f"参考原图与待测图片内容重复: {excluded_overlap}")
        items = [i for i in items if i["image_sha256"] not in hashes]
        if not items or config["mode"] == "train" and all(i["ground_truth"] is None for i in items):
            raise ValueError("排除参考图后没有可处理/可评估的待测图片")
        if excluded_overlap:
            LOGGER.info("排除与参考集重叠的待测图: %s", ", ".join(excluded_overlap))
    validate_reference_separation(config, items)
    from .data.support_manifest import validate_support_selection
    support_manifests = validate_support_selection(config)
    from .result_layout import apply_result_layout
    config = apply_result_layout(config, items, create=not dry_run)
    report = {
        "scene_name": config["scene_name"], "config_file": config["config_file"],
        "mode": config["mode"], "image_dir": str(config["image_dir"]),
        "config_sources": config.get("config_sources", []),
        "dataset_manifest": config.get("dataset_manifest"),
        "dataset_manifest_sha256": image_sha256(Path(config["dataset_manifest"])) if config.get("dataset_manifest") else None,
        "support_manifests": support_manifests,
        "output_dir": str(config["output_dir"]),
        "prompt_index_root": str(config.get('prompt_index_root', config.get('prediction_cache_dir', config['output_dir']))),
        "result_layout": config.get('result_layout'),
        "image_count": len(items), **truth_summary(items),
        "excluded_reference_overlap": excluded_overlap,
        "dataset_sha256": digest({"images": [{"relative_path": i["relative_path"],
            "image_sha256": i["image_sha256"], "ground_truth": i["ground_truth"]} for i in items]}),
        "image_paths": [item["relative_path"] for item in items],
        "ground_truth_dir": str(config["ground_truth_dir"]),
        "missing_ground_truth": config["missing_ground_truth"],
        "evaluation": config["evaluation"],
        "cache_version": config.get("cache_version", "v1"),
        "experiment": config.get("experiment", {}),
        "targets": [{
            "id": t["id"], "backend": t["backend"], "label": t["label"],
            **({'labels': target_label_names(t)} if len(target_label_names(t)) > 1 else {}),
            "model": t["config"]["model"], "base_url": t["config"]["base_url"],
            'api_protocol': t['config'].get('api_protocol', 'openai_chat_completions'),
            **({'proxy_request_model': t['config']['proxy_request_model']}
               if t['config'].get('proxy_request_model') else {}),
            "prompt_file": t["prompt_file"], "review_prompt_file": t["review_prompt_file"],
            "reference_image_dir": t["config"]["reference_image_dir"],
            "reference_mode": t["config"].get("reference_mode", "configured"),
            "prompt_snapshot": capture_prompt(t),
            'require_box_reason': t['config'].get('require_box_reason', False),
            **({"local_model": t["config"]["local_model"],
                "reference_sample_ids": [r["sample_id"] for r in t["config"]["reference_examples"]]}
               if is_local(t["backend"]) else {}),
            "generation_settings": {key: t["config"][key] for key in DEFAULTS} if not is_local(t["backend"]) else {},
            "reference_count": {k: len(t["config"].get(k, [])) for k in ("reference_examples", "reference_crops", "reference_negatives", "reference_originals", "reference_boxed_images")},
            "reference_manifest_sha256": t["config"]["reference_manifest_sha256"],
            "prompt_sha256": hashlib.sha256((t["prompt"] + t["review_template"]).encode()).hexdigest(),
        } for t in config["targets"]],
    }
    reuse_enabled = config.get("reuse_predictions", True) and not force
    caches, reusable = {}, {}
    for target, descriptor in zip(config["targets"], report["targets"]):
        cache = PredictionCache(config.get("prediction_cache_dir", config["output_dir"]), config["image_dir"], descriptor,
                                config.get("cache_version", "v1"), config.get("cache_search_dirs"))
        caches[target["id"]] = cache
        descriptor["request_signature"] = cache.signature
        descriptor['prompt_snapshot']['request_signature'] = cache.signature
        if reuse_enabled:
            for item in items:
                record = cache.get(item)
                if record is not None:
                    reusable[target["id"], item["relative_path"]] = record
    scheduled = len(items) * len(config["targets"]) - len(reusable)
    report.update(reused_prediction_count=len(reusable), scheduled_inference_count=scheduled, reuse_predictions=reuse_enabled)
    LOGGER.info("复用成功结果 %d 张，待请求模型 %d 张", len(reusable), scheduled)
    if reusable and config['mode'] == 'train':
        LOGGER.info("复用的是已保存的检测框；指标将按本次评估参数重新计算（score_threshold=%s，IoU=%s）。",
                    config['evaluation']['score_threshold'], config['evaluation']['iou_threshold'])
    if dry_run:
        planned = result_dir if result_dir is not None else result_directory(config['output_dir'], model_shot_name(config))
        return {**report, "dry_run": True, 'run_dir': str(planned)}
    if scheduled and client_factory is None and any(not is_local(t["backend"]) for t in config["targets"]):
        from .dependencies import load_openai_client
        client_factory = load_openai_client()
    from .credentials import resolve_api_key
    api_keys = {target['id']: resolve_api_key(target['model_config'])
                for target in config['targets'] if not is_local(target['backend'])
                and any((target['id'], item['relative_path']) not in reusable for item in items)}
    local_clients = {}
    for target in local_targets:
        if any((target["id"], i["relative_path"]) not in reusable for i in items):
            local_clients[target["id"]] = await asyncio.to_thread(local_runner_factory or local_runner_class(target["backend"]), target["config"])
    if local_targets:
        report["local_model_setup"] = {key: {
            "model_load_seconds": getattr(client, "setup_seconds", None),
            "reference_encoding_seconds": getattr(client, "reference_encoding_seconds", None),
            **({'text_encoding_seconds': client.text_encoding_seconds}
               if hasattr(client, 'text_encoding_seconds') else {}),
        } for key, client in local_clients.items()}
    if result_dir is not None:
        run_dir = result_dir
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = result_directory(config['output_dir'], model_shot_name(config), create=True)
    LOGGER.info("本次结果保存目录: %s", run_dir)
    write_json(run_dir / "run.json", report)
    if config.get("resolved_config"):
        write_json(run_dir / "config.resolved.json", config["resolved_config"])
    target_dirs = {}
    for target, descriptor in zip(config["targets"], report['targets']):
        target_dir = run_dir if len(config['targets']) == 1 else run_dir / target['id']
        target_dir.mkdir(exist_ok=True)
        target_dirs[target['id']] = target_dir
        save_prompt(target_dir, descriptor['prompt_snapshot'])
        (target_dir / "review_template.txt").write_text(target["review_template"], encoding="utf-8")
    records = []
    for target in config["targets"]:
        from .service_errors import ModelServiceError
        model = target["model_config"]
        client = local_clients.get(target["id"])
        cache = caches[target["id"]]
        queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)
        abort_requests = False

        async def worker():
            nonlocal client, abort_requests
            while not queue.empty() and not abort_requests:
                item = queue.get_nowait()
                lock = cache.lock(item)
                await lock.acquire()
                try:
                    fatal_error = None
                    record = None
                    if reuse_enabled:
                        # Another process may have completed this image after our
                        # initial scan. Recheck under the per-image OS lock.
                        record = cache.get(item)
                        if record is None:
                            record = reusable.get((target["id"], item["relative_path"]))
                    if record is None:
                        if client is None:
                            client = client_factory(
                                base_url=model["base_url"], api_key=api_keys[target['id']],
                                timeout=target["config"]["timeout_seconds"], max_retries=0,
                            )
                        started = time.perf_counter()
                        try:
                            record = await infer_image(client, target, item)
                        except ModelServiceError as exc:
                            abort_requests = True
                            fatal_error = exc
                            record = exc.record or {'relative_path': item['relative_path'],
                                'target_id': target['id'], 'status': 'failed', 'error': str(exc)}
                        record["inference_seconds"] = time.perf_counter() - started
                        record["cache_hit"] = False
                    record["image_sha256"] = item["image_sha256"]
                    if record["status"] == "success":
                        from .data.prediction_outputs import export_prediction_thresholds
                        record.update(export_prediction_thresholds(
                            target_dirs[target['id']], item, record, config["evaluation"]))
                    from .evaluation.feedback import save_feedback
                    record.update(save_feedback(run_dir, record, target, item['width'], item['height'],
                                                target_dir=target_dirs[target['id']]))
                    records.append(record)
                    with (run_dir / "records.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    if record["status"] == "success":
                        try:
                            cache.save(item, record, run_dir)
                        except OSError as exc:
                            LOGGER.warning("写入缓存失败，已保留运行记录供下次恢复: %s", exc)
                    LOGGER.info("%s %d/%d %s: %s%s", target["id"], len(records), len(items) * len(config["targets"]), item["relative_path"], record["status"], "（复用）" if record.get("cache_hit") else "")
                    if fatal_error is not None:
                        raise fatal_error
                finally:
                    lock.release()
                    queue.task_done()
        workers = [asyncio.create_task(worker()) for _ in range(min(config['concurrency'], len(items)))]
        try:
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                if client is not None:
                    try:
                        await asyncio.wait_for(client.close(), timeout=3)
                    except Exception as close_error:
                        LOGGER.warning('关闭模型连接未完成（%s）；继续退出。', type(close_error).__name__)
        except BaseException as exc:
            interrupted = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            report.update(run_dir=str(run_dir), status='interrupted' if interrupted else 'failed',
                          evaluation_status='interrupted' if interrupted else 'not_completed',
                          success_request_count=sum(r['status'] == 'success' for r in records),
                          failed_request_count=sum(r['status'] != 'success' for r in records))
            write_json(run_dir / 'summary.json', {**report, 'annotations': records})
            raise
    records.sort(key=lambda r: (r["relative_path"], r["target_id"]))
    failed = [r for r in records if r["status"] != "success"]
    report.update(run_dir=str(run_dir), failed_request_count=len(failed), success_request_count=len(records) - len(failed))
    report["reused_prediction_count"] = sum(bool(record.get("cache_hit")) for record in records)
    report["executed_inference_count"] = len(records) - report["reused_prediction_count"]
    timed = [r["inference_seconds"] for r in records if not r.get("cache_hit") and "inference_seconds" in r]
    report["timing"] = {"fresh_timed_image_count": len(timed), "fresh_inference_seconds": sum(timed),
                        "mean_fresh_image_seconds": sum(timed) / len(timed) if timed else None,
                        "policy": "Per-image wall time includes preprocessing and retries; excludes setup/cache hits. "
                                  "Sum of concurrent request latencies is not end-to-end throughput."}
    report["evaluation_status"] = "pending" if config["mode"] == "train" else "not_requested"
    # Persist the complete inference summary before optional metric calculation.
    write_json(run_dir / "summary.json", {**report, "annotations": records})
    if config["mode"] == "train":
        try:
            metrics = evaluate_records(items, records, all_label_names(config['targets']), config["evaluation"])
        except Exception as exc:
            report.update(evaluation_status="failed", evaluation_error=str(exc))
            write_json(run_dir / "summary.json", {**report, "annotations": records})
            raise RuntimeError(
                f"推理结果已保存到 {run_dir}，评估失败: {exc}。"
                '修复后重新运行原命令即可自动复用成功结果，无需重跑推理。'
            ) from exc
        write_json(run_dir / "metrics.json", metrics)
        report["metrics"] = metrics
        report["evaluation_status"] = metrics["evaluation_status"]
        from .evaluation.reporting import save_text_report
        save_text_report(run_dir, report)
    write_json(run_dir / "summary.json", {**report, "annotations": records})
    index_roots = {Path(report['prompt_index_root'])}
    if config.get('result_layout'):
        index_roots.update(Path(config['result_layout'][key]) for key in ('dataset_dir', 'prompt_dir'))
    for index_root in sorted(index_roots):
        update_prompt_index(index_root)
    return report





from .data.dataset import inspect_dataset
from .cli import main
