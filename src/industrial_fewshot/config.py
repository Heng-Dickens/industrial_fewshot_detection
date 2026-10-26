"""Read any independent scene config and render its prompt files."""
from __future__ import annotations

from pathlib import Path
import json
import math

from industrial_fewshot.data.labelme import MISSING_GT_POLICIES, read_json
from industrial_fewshot.evaluation.detection_metrics import evaluation_settings
from industrial_fewshot.prompts.references import REFERENCE_MODES, configure_reference_inputs

SUPPORTED_BACKENDS = {"generic", "diaowu", "diaogou", "yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"}
from .paths import PROJECT_ROOT, TEMPLATES_DIR, resolve_location
from .configuration import load_document, redact


DEFAULTS = {
    "concurrency": 4, "max_attempts": 3, "retry_delay_seconds": 2.0,
    "timeout_seconds": 180.0, "max_tokens": 2048, "temperature": 0.1,
    "top_p": 0.9, "enable_thinking": False, "stream": False,
    "enable_completeness_review": True, "structured_output_mode": "json_schema",
}



def resolve(base: Path, value: str) -> Path:
    return (base / Path(value).expanduser()).resolve()



def render_prompt(text: str, target: dict, initial: dict | None = None) -> str:
    labels = target.get('labels', [{'name': target['label'], 'description': target['description']}])
    values = {
        "label_name": target["label"], "label_description": target["description"],
        "label_lines": '\n'.join(f'- `{row["name"]}`: {row["description"]}' for row in labels),
    }
    if initial is not None:
        values["initial_result"] = json.dumps(initial, ensure_ascii=False)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text



def load_config(path: Path, mode: str | None = None, image_dir: Path | None = None,
                output_dir: Path | None = None) -> dict:
    path = resolve_location(path)
    raw, config_sources = load_document(path)
    base = path.parent
    mode = mode or raw.get("mode", "test")
    if mode not in {"train", "test"}:
        raise ValueError("mode 只能是 train 或 test")
    if "active_targets" in raw or "targets" in raw:
        raise ValueError("请使用独立场景配置：配置 backend、labels、reference_image_dir 和 prompt_file，不再使用目标总表")
    dataset = raw.get("dataset", raw)
    source = image_dir.expanduser().resolve() if image_dir else resolve(base, dataset["image_dir"])
    if not source.is_dir():
        raise ValueError(f"待标注图片目录不存在，请设置 dataset.image_dir 或 --image-dir: {source}")
    labels = raw.get("labels")
    if (
        not isinstance(labels, list) or not labels
        or any(not isinstance(row, dict)
            or not isinstance(row.get('name'), str) or not row['name'].strip()
            or not isinstance(row.get('description'), str) or not row['description'].strip()
            for row in labels)
    ):
        raise ValueError("场景配置 labels 必须包含非空 name/description 目标类别")
    if len({row['name'] for row in labels}) != len(labels):
        raise ValueError('labels 包含重复类别名')
    if len(labels) > 1 and raw.get('backend') != 'generic':
        raise ValueError('多类别单次请求目前仅支持 generic MLLM 后端')
    # Accept the existing flat per-scene configs; a nested inference section may
    # optionally override their generation settings.
    inference = {**DEFAULTS, **{k: raw[k] for k in DEFAULTS if k in raw}, **raw.get("inference", {})}
    from .detectors.registry import is_local
    local_detector = is_local(raw.get('backend'))
    if local_detector:
        if raw['backend'] in {'pet_dino', 'wedetect'}:
            from .detectors.prompt_native import native_settings
            local_settings = native_settings(raw['backend'], raw.get(raw['backend'], {}), base)
        elif raw["backend"] == "yoloe":
            from industrial_fewshot.detectors.yoloe_visual import yoloe_settings
            local_settings = yoloe_settings(raw.get("yoloe", {}), base)
        elif raw['backend'] == 'yoloe_fusion':
            from .detectors.yoloe_fusion import yoloe_fusion_settings
            local_settings = yoloe_fusion_settings(raw.get('yoloe_fusion', {}), base)
        elif raw['backend'] == 'yoloe_text':
            from .detectors.yoloe_text import yoloe_text_settings
            local_settings = yoloe_text_settings(raw.get('yoloe_text', {}), base)
        elif raw['backend'] == 'groundingdino':
            from industrial_fewshot.detectors.groundingdino_text import groundingdino_settings
            local_settings = groundingdino_settings(raw.get("groundingdino", {}), base)
        else:
            from .detectors.yolo_onnx import yolo_onnx_settings
            local_settings = yolo_onnx_settings(raw.get('yolo_onnx', {}), base)
        inference.update(concurrency=1, max_attempts=1, enable_completeness_review=False)
    for key in ("concurrency", "max_attempts", "max_tokens"):
        if isinstance(inference[key], bool) or not isinstance(inference[key], int) or inference[key] < 1:
            raise ValueError(f"inference.{key} 必须是正整数")
    for key in ("timeout_seconds", "retry_delay_seconds", "temperature", "top_p"):
        value = inference[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"inference.{key} 必须是有限非负数字")
    if inference["timeout_seconds"] <= 0 or not 0 < inference["top_p"] <= 1 or inference["temperature"] > 2:
        raise ValueError("timeout_seconds > 0、0 < top_p <= 1、0 <= temperature <= 2")
    for key in ("enable_thinking", "stream", "enable_completeness_review"):
        if not isinstance(inference[key], bool):
            raise ValueError(f"inference.{key} 必须是布尔值")
    if inference["structured_output_mode"] not in {"json_schema", "json_object", "none"}:
        raise ValueError("structured_output_mode 无效")
    target_id = raw.get("task_id", raw.get("backend"))
    if not isinstance(target_id, str) or not target_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in target_id):
        raise ValueError("目标 ID 只能包含字母、数字、下划线、短横线")
    target = dict(raw)
    target.update(id=target_id, label=labels[0]["name"], description=labels[0]["description"])
    if target.get("backend") not in SUPPORTED_BACKENDS:
        raise ValueError(f"未知标注子程序: {target.get('backend')}")
    if not target.get("label") or not target.get("description"):
        raise ValueError(f"{target_id} 必须配置 label 和 description")
    if local_detector:
        model = {"name": local_settings.get("weights", local_settings.get("checkpoint")), "base_url": None}
    elif isinstance(raw.get("model"), str):
        model = {k: raw[k] for k in ("base_url", "api_key_env", "api_key", "api_key_file", "env_file", "api_protocol", "proxy_request_model") if k in raw}
        model["name"] = raw["model"]
    elif isinstance(raw.get("model"), dict):
        model = dict(raw["model"])
    else:
        raise ValueError("model 必须是模型名称字符串或包含 name/base_url 的对象")
    if not model.get("name") or (not local_detector and not model.get("base_url")):
        raise ValueError("model 必须配置 name 和 base_url")
    protocol = model.get('api_protocol', 'openai_chat_completions')
    proxy_model = model.get('proxy_request_model')
    if proxy_model is not None and (not isinstance(proxy_model, str) or not proxy_model.strip()
                                   or protocol != 'anthropic_messages'):
        raise ValueError('proxy_request_model 必须是非空Claude代理路由名，且协议为anthropic_messages')
    if not local_detector and protocol not in {'openai_chat_completions', 'anthropic_messages'}:
        raise ValueError('api_protocol 必须是 openai_chat_completions 或 anthropic_messages')
    if not local_detector and protocol == 'anthropic_messages' and inference['enable_thinking']:
        raise ValueError('当前 Claude 转发适配请使用 enable_thinking: false')
    reference = resolve(base, target["reference_image_dir"]) if target.get("reference_image_dir") else None
    prompt_file = resolve(base, target["prompt_file"]) if not local_detector else None
    target["prompt"] = render_prompt(prompt_file.read_text(encoding="utf-8-sig"), target) if prompt_file else ""
    if not local_detector and not target["prompt"].strip():
        raise ValueError(f"提示词文件为空: {prompt_file}")
    review_file = resolve(base, target["review_prompt_file"]) if target.get("review_prompt_file") else TEMPLATES_DIR / "review.txt"
    target["review_template"] = review_file.read_text(encoding="utf-8-sig") if inference["enable_completeness_review"] else ""
    config = {
        **inference, "model": model["name"], "base_url": model["base_url"], 'api_protocol': protocol,
        **({'proxy_request_model': proxy_model} if proxy_model is not None else {}),
        "labels": [dict(row) for row in labels],
        "reference_per_class": raw.get('reference_per_class', False),
        "reference_image_dir": str(reference) if reference else None, "require_confidence": True,
        "support_manifest_dir": raw.get('support_manifest_dir'),
    }
    for key, default in (("max_reference_examples", 6), ("max_reference_crops", 0 if target["backend"] == "diaogou" else 6), ("max_reference_negatives", 0)):
        if target['backend'] in {'yolo_onnx', 'yoloe_text', 'yoloe_fusion', 'pet_dino', 'wedetect'}:
            default = 0
        value = target.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{target_id}.{key} 必须是非负整数")
        config[key] = value
        if target['backend'] in {'yolo_onnx', 'yoloe_text', 'yoloe_fusion', 'pet_dino', 'wedetect'} and value:
            raise ValueError('该检测模式不接收参考提示，请将 max_reference_* 设置为 0')
    if local_detector:
        reference_mode = {'yoloe': 'yoloe_visual', 'yoloe_text': 'yoloe_text', 'yoloe_fusion': 'yoloe_fusion', 'groundingdino': 'ovd_text', 'yolo_onnx': 'supervised_onnx', 'pet_dino': 'pet_dino', 'wedetect': 'wedetect'}[raw['backend']]
        config.update({raw["backend"]: local_settings}, reference_mode=reference_mode)
    else:
        configure_reference_inputs(config, raw.get("reference_mode", "configured"))
    if any(config[key] for key in ("max_reference_examples", "max_reference_crops", "max_reference_negatives")):
        if reference is None or not reference.is_dir():
            raise ValueError(f"参考图目录不存在: {reference}")
    from industrial_fewshot.data.supports import suite_settings
    experiment_suite = suite_settings(raw.get("experiment_suite"))
    ablation = raw.get("ablation", {})
    if not isinstance(ablation, dict) or not isinstance(ablation.get("enabled", False), bool):
        raise ValueError("ablation 必须是配置对象，enabled 必须是布尔值")
    modes = ablation.get("modes", list(REFERENCE_MODES))
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(value, str) or value not in REFERENCE_MODES for value in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError(f"ablation.modes 必须是不重复的输入组合列表: {REFERENCE_MODES}")
    if local_detector and (experiment_suite["enabled"] or ablation.get("enabled", False)):
        raise ValueError("本地检测器不使用 VLM experiment_suite/ablation，请关闭这两个开关")
    gate = target.get("require_suspension_evidence", False)
    if not isinstance(gate, bool):
        raise ValueError("require_suspension_evidence 必须是布尔值")
    if gate and target["backend"] != "diaowu":
        raise ValueError("悬吊确认字段仅用于 diaowu 子程序")
    config["require_suspension_evidence"] = gate
    box_reason = raw.get('require_box_reason', not local_detector)
    if not isinstance(box_reason, bool):
        raise ValueError('require_box_reason 必须是布尔值')
    if local_detector and box_reason:
        raise ValueError('本地检测器不输出自然语言框理由，请设置 require_box_reason=false')
    config['require_box_reason'] = box_reason
    if box_reason:
        target['prompt'] += (
            '\n【逐框反馈】每个 objects 元素必须额外包含非空字符串 reason。'
            '用一两句简洁中文说明该目标的可见特征、符合目标定义的证据，以及框为什么覆盖这些部位。'
            '只描述图像中可观察的依据和框的覆盖范围，不输出详细推理过程，不编造不可见细节。'
            '每个框单独填写，与该框对应；复核修改框后同步更新 reason。'
            '整图的 reason 继续说明总体判断；没有目标或无法判断时说明相应原因。'
        )
    if not local_detector:
        target["prompt"] += (
        '\n【统一输出要求】每个 objects 元素还必须包含 "confidence": 0~1 的有限数字，'
        '表示该框属于指定目标的检测置信度；不要省略。该要求补充前面的 JSON 示例。'
        )
    if gate:
        target["prompt"] += '\n每个框还必须包含布尔字段 hanging_object_confirmed，确认框内有符合条件的被吊物本体。'
    target.update(config=config, model_config=model, prompt_file=str(prompt_file) if prompt_file else None,
                  review_prompt_file=str(review_file) if not local_detector else None)
    configured_targets = [target]
    suffix = raw.get("output_suffix", "_annotations")
    if (not isinstance(suffix, str) or not suffix.strip() or suffix.endswith((".", " "))
            or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in suffix)):
        raise ValueError("output_suffix 必须是非空文件夹名称后缀，不能包含路径分隔符或非法字符")
    specified_output = raw.get("output_dir")
    if specified_output is not None and not isinstance(specified_output, str):
        raise ValueError("output_dir 必须是路径字符串或 null")
    if output_dir is not None:
        output = output_dir.expanduser().resolve()
    elif specified_output and specified_output.strip():
        output = resolve(base, specified_output)
    else:
        if not source.name:
            raise ValueError("图片目录为磁盘根目录时，请显式指定 output_dir")
        output = source.with_name(source.name + suffix).resolve()
    # Reading old locations keeps switching to the sibling default from costing
    # another model request. All writes and locks remain in the selected output.
    cache_search_dirs = [resolve(base, "../runs"), PROJECT_ROOT / "outputs/runs/legacy"]
    cache_search_dirs.extend(PROJECT_ROOT / 'outputs/runs/qwen_mllm' / kind
                             for kind in ('ablation', 'fewshot', 'text_only'))
    if raw.get('backend') == 'yoloe':
        cache_search_dirs.append(PROJECT_ROOT / 'outputs/runs/yoloe_visual' / f"{local_settings['positive_count']}shot")
    if source.name:
        cache_search_dirs.append(source.with_name(source.name + suffix))
    if specified_output and specified_output.strip():
        cache_search_dirs.append(resolve(base, specified_output))
    scene_name = raw.get("scene_name", path.stem)
    if isinstance(scene_name, str) and scene_name not in {"", ".", ".."} and not any(c in scene_name for c in '/\\:'):
        cache_search_dirs.append(PROJECT_ROOT / "runs" / scene_name)
    gt_root = resolve(base, dataset["ground_truth_dir"]) if dataset.get("ground_truth_dir") else source
    for protected in (source, gt_root, *([reference] if reference else [])):
        if output == protected or output.is_relative_to(protected):
            raise ValueError(f"output_dir 必须位于图片/真值/参考目录之外: {output}")
    missing = dataset.get("missing_ground_truth", "skip")
    if missing not in MISSING_GT_POLICIES:
        raise ValueError("missing_ground_truth 只能是 skip、error 或 negative")
    recursive = dataset.get("recursive", True)
    if not isinstance(recursive, bool):
        raise ValueError("dataset.recursive 必须是布尔值")
    extensions = dataset.get("extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    if not isinstance(extensions, list) or not extensions or any(not isinstance(e, str) or not e for e in extensions):
        raise ValueError("dataset.extensions 必须是非空扩展名数组")
    reuse_predictions = raw.get("reuse_predictions", True)
    if not isinstance(reuse_predictions, bool):
        raise ValueError("reuse_predictions 必须是布尔值")
    cache_version = raw.get("cache_version", "v1")
    if not isinstance(cache_version, str) or not cache_version.strip():
        raise ValueError("cache_version 必须是非空字符串")
    if 'results_layout' in raw and not isinstance(raw['results_layout'], bool):
        raise ValueError('results_layout 必须是布尔值')
    return {
        "scene_name": scene_name,
        "results_layout": raw.get("results_layout", output.is_relative_to(PROJECT_ROOT / 'outputs/runs')),
        "resume_comparison_dir": resolve(base, raw['resume_comparison_dir']) if raw.get('resume_comparison_dir') else None,
        "config_sources": config_sources,
        "resolved_config": redact(raw),
        "dataset_manifest": raw.get("dataset", {}).get("manifest_file"),
        "dataset_preparation": raw.get("preparation", {}),
        "support_manifest_dir": raw.get("support_manifest_dir"),
        "mode": mode, "image_dir": source, "ground_truth_dir": gt_root,
        "output_dir": output, "output_suffix": suffix, "cache_search_dirs": cache_search_dirs,
        "targets": configured_targets, "recursive": recursive,
        "missing_ground_truth": missing, "evaluation": evaluation_settings(raw.get("evaluation")),
        "extensions": {"." + e.lower().lstrip(".") for e in extensions}, "config_file": str(path),
        "concurrency": inference["concurrency"],
        "reuse_predictions": reuse_predictions, "cache_version": cache_version,
        "ablation": {"enabled": ablation.get("enabled", False), "modes": modes},
        "experiment_suite": experiment_suite,
    }

