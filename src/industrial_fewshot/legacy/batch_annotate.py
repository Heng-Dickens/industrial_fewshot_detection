#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 OpenAI 兼容的 vLLM 服务批量生成图片辅助标注。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from industrial_fewshot.postprocess.schema import build_json_schema, build_response_format, clean_json_response, validate_prediction, ANNOTATION_STATUSES, REASON_CODES, UNABLE_REASON_CODES
from industrial_fewshot.detectors.transport import get_image_info, image_mime_type, usage_as_dict, read_completion

LOGGER = logging.getLogger("batch_annotate")
SCRIPT_DIR = Path(__file__).resolve().parent
# 通用程序放在 tool/ 下，配置统一集中在上一层的 config/ 目录
from industrial_fewshot.paths import LEGACY_CONFIG_DIR, resolve_location
CONFIG_DIR = LEGACY_CONFIG_DIR

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量调用 Qwen3-VL/vLLM 生成目标框辅助标注。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "annotation_config.json",
        help="配置文件地址，默认使用 configs/specific/annotation_config.json",
    )
    parser.add_argument("--image-dir", type=Path, help="覆盖配置中的图片目录")
    parser.add_argument("--output", type=Path, help="覆盖最终标注 JSON 地址")
    parser.add_argument("--checkpoint", type=Path, help="覆盖 JSONL 检查点地址")
    parser.add_argument("--concurrency", type=int, help="覆盖并发 worker 数")
    parser.add_argument("--limit", type=int, help="本次最多处理多少张图片")
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新处理已有成功记录；旧检查点不会删除，汇总时使用最新记录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查配置并扫描图片，不调用模型、不写结果",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"配置文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("配置文件顶层必须是 JSON object")
    return value


def resolve_path(config_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return resolve_location(path)


def dataset_identifier(image_dir: Path) -> str:
    """从图片根目录最后一级名称生成安全的任务标号。"""
    raw_name = image_dir.name.strip()
    identifier = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_name).strip("._-")
    if not identifier:
        raise ValueError(f"无法从图片目录提取有效标号: {image_dir}")
    return identifier


def append_dataset_suffix(path: Path, identifier: str) -> Path:
    """在JSON/JSONL文件名中加入目录标号，并避免重复追加。"""
    checkpoint_tail = ".checkpoint.jsonl"
    if path.name.lower().endswith(checkpoint_tail):
        prefix = path.name[: -len(checkpoint_tail)]
        if prefix.endswith(f"_{identifier}"):
            return path
        return path.with_name(f"{prefix}_{identifier}{checkpoint_tail}")
    if path.stem.endswith(f"_{identifier}"):
        return path
    return path.with_name(f"{path.stem}_{identifier}{path.suffix}")


def config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} 必须是 JSON boolean（true 或 false）")
    return value


def config_float(
    config: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是数字")
    result = float(value)
    minimum_ok = result >= minimum if minimum_inclusive else result > minimum
    if not minimum_ok or result > maximum:
        left = "[" if minimum_inclusive else "("
        raise ValueError(f"{key} 必须位于 {left}{minimum}, {maximum}] 范围内")
    return result


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_location(args.config)
    config = read_json(config_path)
    config_dir = config_path.parent

    # 命令行覆盖项按当前工作目录解析；只有配置文件里的相对路径才以配置目录为基准
    if args.image_dir is not None:
        config["image_dir"] = str(resolve_path(Path.cwd(), args.image_dir))
    if args.output is not None:
        config["output_json"] = str(resolve_path(Path.cwd(), args.output))
    if args.checkpoint is not None:
        config["checkpoint_jsonl"] = str(resolve_path(Path.cwd(), args.checkpoint))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency

    required = [
        "base_url",
        "model",
        "image_dir",
        "output_json",
        "checkpoint_jsonl",
        "labels",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段: {', '.join(missing)}")

    labels = config["labels"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels 必须是非空数组")
    label_names: list[str] = []
    for index, item in enumerate(labels):
        if not isinstance(item, dict) or not item.get("name") or not item.get("description"):
            raise ValueError(
                f"labels[{index}] 必须包含非空的 name 和 description"
            )
        label_names.append(str(item["name"]))
    if len(label_names) != len(set(label_names)):
        raise ValueError("labels 中的 name 不能重复")

    concurrency = int(config.get("concurrency", 8))
    max_attempts = int(config.get("max_attempts", 3))
    if concurrency < 1:
        raise ValueError("concurrency 必须大于 0")
    if max_attempts < 1:
        raise ValueError("max_attempts 必须大于 0")

    image_dir_path = resolve_path(config_dir, config["image_dir"])
    output_path = resolve_path(config_dir, config["output_json"])
    checkpoint_path = resolve_path(config_dir, config["checkpoint_jsonl"])
    append_image_dir_suffix = config_bool(
        config, "append_image_dir_suffix", False
    )
    identifier = (
        dataset_identifier(image_dir_path) if append_image_dir_suffix else None
    )
    if append_image_dir_suffix:
        assert identifier is not None
        output_path = append_dataset_suffix(output_path, identifier)
        checkpoint_path = append_dataset_suffix(checkpoint_path, identifier)

    config["config_path"] = str(config_path)
    config["image_dir"] = str(image_dir_path)
    config["output_json"] = str(output_path)
    config["checkpoint_jsonl"] = str(checkpoint_path)
    config["append_image_dir_suffix"] = append_image_dir_suffix
    config["dataset_id"] = identifier
    config["concurrency"] = concurrency
    config["max_attempts"] = max_attempts
    config["timeout_seconds"] = float(config.get("timeout_seconds", 180))
    config["retry_delay_seconds"] = float(config.get("retry_delay_seconds", 2))
    config["max_tokens"] = int(config.get("max_tokens", 512))
    if config["max_tokens"] < 1:
        raise ValueError("max_tokens 必须大于 0")
    config["temperature"] = config_float(config, "temperature", 0.0, 0.0, 2.0)
    config["top_p"] = config_float(
        config, "top_p", 1.0, 0.0, 1.0, minimum_inclusive=False
    )
    config["enable_thinking"] = config_bool(config, "enable_thinking", False)
    config["stream"] = config_bool(config, "stream", False)
    config["enable_completeness_review"] = config_bool(
        config, "enable_completeness_review", False
    )
    config["prompt_version"] = str(config.get("prompt_version", "v1"))
    config["structured_output_mode"] = str(
        config.get("structured_output_mode", "json_schema")
    )
    if config["structured_output_mode"] not in {"json_schema", "json_object", "none"}:
        raise ValueError(
            "structured_output_mode 只能是 json_schema、json_object 或 none"
        )

    extensions = config.get(
        "extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("extensions 必须是非空数组")
    config["extensions"] = {
        extension.lower()
        if str(extension).startswith(".")
        else f".{str(extension).lower()}"
        for extension in extensions
    }
    return config


def scan_images(image_dir: Path, extensions: set[str]) -> list[Path]:
    if not image_dir.exists():
        raise ValueError(f"图片目录不存在: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"图片地址不是目录: {image_dir}")
    return sorted(
        path.resolve()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def make_image_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def build_prompt(config: dict[str, Any]) -> str:
    label_lines = "\n".join(
        f'- `{item["name"]}`: {item["description"]}' for item in config["labels"]
    )
    custom_instruction = str(config.get("extra_instruction", "")).strip()
    prompt = f"""你是一个严谨的目标检测辅助标注器，
用来检测不同类型的车辆目标，
请检测图片中所有属于以下标签的可见目标：
{label_lines}

标注要求：
1. 先从左到右、从上到下完整扫描整张图片，再输出结果。属于允许标签的每一个独立物体实例都必须分别输出一个紧致边界框，不要包含明显无关背景。
2. 同类目标有多个时不得合并、不得只标最大/最近/最显眼的一个，也不得因为已经找到一个目标就停止搜索。
3. target_count 必须填写本图最终输出的目标实例总数，并且必须严格等于 objects 数组长度；没有目标或无法判定时均为 0。
4. bbox_2d 格式固定为 [x1, y1, x2, y2]。
5. 坐标是 0~1000 的相对坐标；左上角为 [0, 0]，右下角为 [1000, 1000]。
6. 只能使用上面给出的标签名，不要创建新标签。
7. 不存在目标时 objects 必须返回空数组，并将 annotation_status 设为 no_target。
8. 标注的前提是图片清晰可见，仅当车辆区域清晰、可正常观察时才允许进行判断。
    以下情况 objects 返回空数组，annotation_status 设为 unable_to_determine：
    - 图像模糊、失焦、低分辨率
    - 光线过暗、过曝、逆光
    - 车辆超过三分之一被遮挡（如设备、其他人员遮挡）
    - 车辆距离过远，占图像总面积小于0.5%
9. 每张图片都必须返回 annotation_status、reason_code 和非空 reason：
    - 检测到目标：annotation_status=annotated，reason_code=target_detected。
    - 图像清晰但没有指定目标：annotation_status=no_target，reason_code=no_target。
    - 无法可靠判断：annotation_status=unable_to_determine，并使用最主要的原因代码：
      blurry、low_resolution、too_dark、overexposed、backlit、occluded、target_too_small 或 other。
10. reason 使用简洁中文说明判断依据；即使 objects 为空也不能省略 reason。
11. 严格按指定 JSON Schema 输出，不要输出解释、Markdown 或代码块。
"""
    if custom_instruction:
        prompt += f"\n补充要求：\n{custom_instruction}\n"
    return prompt


def build_completeness_review_prompt(
    config: dict[str, Any], initial_result: dict[str, Any]
) -> str:
    """构造第二遍完整性复核提示词，要求返回完整最终结果而非增量。"""
    base_prompt = build_prompt(config)
    initial_for_model = {
        "annotation_status": initial_result["annotation_status"],
        "target_count": initial_result["target_count"],
        "reason_code": initial_result["reason_code"],
        "reason": initial_result["reason"],
        "objects": [
            {"label": item["label"], "bbox_2d": item["bbox_2d"]}
            for item in initial_result["objects"]
        ],
    }
    return f"""{base_prompt}

你现在正在执行同一张图片的第二遍完整性复核。

第一遍结果如下：
{json.dumps(initial_for_model, ensure_ascii=False)}

请忽略第一遍已经找到几个目标，重新从左到右、从上到下独立扫描整张图片：
1. 检查是否遗漏了同类的第二个、第三个或更多实例，特别是画面边缘、远处和相互靠近的目标。
2. 每个独立物体必须单独输出一个框；不能合并多个物体，不能只保留最大、最近或最明显的一个。
3. 同时删除明显误检并修正不紧致的框。
4. 输出必须是复核后的完整最终列表，不能只输出相对第一遍新增的目标。
5. target_count 必须等于最终 objects 数组长度。
6. 继续遵守图像质量、遮挡、最小面积、标签范围及 0~1000 坐标要求。
7. 严格按指定 JSON Schema 输出，不要输出解释、Markdown 或代码块。
"""














def relative_image_path(image_path: Path, image_dir: Path) -> str:
    return image_path.relative_to(image_dir).as_posix()


def load_latest_records(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return latest
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("忽略检查点第 %d 行的损坏记录", line_number)
                continue
            relative_path = record.get("relative_path")
            if isinstance(relative_path, str):
                latest[relative_path] = record
    return latest


class CheckpointWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = asyncio.Lock()

    async def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self.lock:
            await asyncio.to_thread(self._append_sync, line)

    def _append_sync(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()


def generation_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "enable_thinking": config["enable_thinking"],
        "stream": config["stream"],
    }


def base_record(image_path: Path, image_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    relative_path = relative_image_path(image_path, image_dir)
    return {
        "image_id": make_image_id(relative_path),
        "relative_path": relative_path,
        "image_path": str(image_path),
        "file_name": image_path.name,
        "dataset_id": config.get("dataset_id"),
        "model": config["model"],
        "prompt_version": config["prompt_version"],
        "generation_settings": generation_settings(config),
    }






async def request_annotation(
    client: Any,
    config: dict[str, Any],
    prompt: str,
    image_data_url: str,
    response_format: dict[str, Any] | None,
    label_names: set[str],
    width: int,
    height: int,
) -> dict[str, Any]:
    """发送一次结构化标注请求并完成业务校验。"""
    request: dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "stream": config["stream"],
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": config["enable_thinking"],
            }
        },
    }
    if config["stream"]:
        request["stream_options"] = {"include_usage": True}
    if response_format is not None:
        request["response_format"] = response_format

    response = await client.chat.completions.create(**request)
    content, finish_reason, usage_dict, reasoning_content = await read_completion(
        response, config["stream"]
    )
    if not content.strip():
        reasoning_length = len(reasoning_content or "")
        raise ValueError(
            "模型响应内容为空"
            f"（finish_reason={finish_reason!r}, "
            f"reasoning_content字符数={reasoning_length}）"
        )
    prediction = clean_json_response(content)
    annotation_result = validate_prediction(
        prediction, label_names, width, height
    )
    return {
        "annotation": annotation_result,
        "raw_response": content,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "usage": usage_dict,
    }


async def annotate_one(
    client: Any,
    image_path: Path,
    image_dir: Path,
    config: dict[str, Any],
    prompt: str,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    record = base_record(image_path, image_dir, config)
    started = time.perf_counter()

    try:
        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        width, height, image_format = await asyncio.to_thread(
            get_image_info, image_bytes
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error_type": "image_error",
                "error": str(exc),
                "attempts": 0,
                "completed_at": now_iso(),
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        )
        return record

    record.update(
        {
            "width": width,
            "height": height,
            "image_format": image_format,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
    )
    mime_type = image_mime_type(image_path, image_format)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:{mime_type};base64,{image_base64}"
    label_names = {str(item["name"]) for item in config["labels"]}
    last_error: Exception | None = None

    for attempt in range(1, config["max_attempts"] + 1):
        try:
            initial = await request_annotation(
                client,
                config,
                prompt,
                image_data_url,
                response_format,
                label_names,
                width,
                height,
            )
            final = initial
            review_attempted = False
            review_performed = False
            review_error: str | None = None
            if config.get("enable_completeness_review", False):
                review_attempted = True
                review_prompt = build_completeness_review_prompt(
                    config, initial["annotation"]
                )
                try:
                    final = await request_annotation(
                        client,
                        config,
                        review_prompt,
                        image_data_url,
                        response_format,
                        label_names,
                        width,
                        height,
                    )
                    review_performed = True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    review_error = str(exc)
                    LOGGER.warning(
                        "完整性复核失败，保留第一遍有效结果: %s: %s",
                        record["relative_path"],
                        exc,
                    )

            initial_annotation = initial["annotation"]
            final_annotation = final["annotation"]
            record.update(
                {
                    "status": "success",
                    **final_annotation,
                    "initial_annotation": initial_annotation,
                    "initial_raw_response": initial["raw_response"],
                    "initial_reasoning_content": initial["reasoning_content"],
                    "initial_finish_reason": initial["finish_reason"],
                    "initial_usage": initial["usage"],
                    "review_attempted": review_attempted,
                    "review_performed": review_performed,
                    "review_changed": (
                        review_performed and initial_annotation != final_annotation
                    ),
                    "review_changed_target_count": (
                        review_performed
                        and initial_annotation["target_count"]
                        != final_annotation["target_count"]
                    ),
                    "review_error": review_error,
                    "raw_response": final["raw_response"],
                    "reasoning_content": final["reasoning_content"],
                    "finish_reason": final["finish_reason"],
                    "usage": final["usage"],
                    "attempts": attempt,
                    "completed_at": now_iso(),
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            return record
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "标注失败，准备重试 (%d/%d): %s: %s",
                attempt,
                config["max_attempts"],
                record["relative_path"],
                exc,
            )
            if attempt < config["max_attempts"]:
                await asyncio.sleep(config["retry_delay_seconds"] * (2 ** (attempt - 1)))

    record.update(
        {
            "status": "failed",
            "error_type": "inference_error",
            "error": str(last_error) if last_error else "unknown error",
            "attempts": config["max_attempts"],
            "completed_at": now_iso(),
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
    )
    return record


async def worker(
    worker_id: int,
    queue: asyncio.Queue[Path | None],
    client: Any,
    writer: CheckpointWriter,
    image_dir: Path,
    config: dict[str, Any],
    prompt: str,
    response_format: dict[str, Any] | None,
    counters: dict[str, int],
) -> None:
    while True:
        image_path = await queue.get()
        try:
            if image_path is None:
                return
            record = await annotate_one(
                client,
                image_path,
                image_dir,
                config,
                prompt,
                response_format,
            )
            await writer.append(record)
            counters[record["status"]] += 1
            counters["processed"] += 1
            LOGGER.info(
                "[%d] %d/%d %s - %s",
                worker_id,
                counters["processed"],
                counters["scheduled"],
                record["status"],
                record["relative_path"],
            )
        finally:
            queue.task_done()


def build_summary(
    config: dict[str, Any],
    image_count: int,
    scheduled_count: int,
) -> dict[str, Any]:
    output_path = Path(config["output_json"])
    checkpoint_path = Path(config["checkpoint_jsonl"])
    latest = load_latest_records(checkpoint_path)
    annotations = sorted(latest.values(), key=lambda item: item.get("relative_path", ""))
    success_count = sum(item.get("status") == "success" for item in annotations)
    failed_count = sum(item.get("status") == "failed" for item in annotations)
    object_count = sum(
        len(item.get("objects", []))
        for item in annotations
        if item.get("status") == "success"
    )
    annotation_status_counts = Counter(
        item.get("annotation_status", "missing")
        for item in annotations
        if item.get("status") == "success"
    )
    no_annotation_reason_counts = Counter(
        item.get("reason_code", "missing")
        for item in annotations
        if item.get("status") == "success" and not item.get("objects")
    )
    current_version_count = sum(
        is_reusable_success(item, config) for item in annotations
    )
    return {
        "format_version": "1.3",
        "generated_at": now_iso(),
        "dataset": {
            "dataset_id": config.get("dataset_id"),
            "image_root": config["image_dir"],
            "annotation_file": str(output_path),
            "checkpoint_file": str(checkpoint_path),
            "discovered_image_count": image_count,
            "scheduled_this_run": scheduled_count,
        },
        "inference": {
            "base_url": config["base_url"],
            "model": config["model"],
            "prompt_version": config["prompt_version"],
            "generation_settings": generation_settings(config),
            "enable_completeness_review": config["enable_completeness_review"],
            "coordinate_system": "relative_0_1000_xyxy",
            "labels": config["labels"],
        },
        "statistics": {
            "record_count": len(annotations),
            "success_count": success_count,
            "failed_count": failed_count,
            "object_count": object_count,
            "current_prompt_version_count": current_version_count,
            "outdated_or_incomplete_count": len(annotations) - current_version_count,
            "annotation_status_counts": dict(sorted(annotation_status_counts.items())),
            "no_annotation_reason_counts": dict(
                sorted(no_annotation_reason_counts.items())
            ),
        },
        "annotations": annotations,
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    temp_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, output_path)


def is_reusable_success(record: dict[str, Any], config: dict[str, Any]) -> bool:
    """判断检查点记录能否用于当前模型和提示词版本。"""
    objects = record.get("objects")
    target_count = record.get("target_count")
    return (
        record.get("status") == "success"
        and record.get("model") == config["model"]
        and record.get("prompt_version") == config["prompt_version"]
        and record.get("generation_settings") == generation_settings(config)
        and record.get("annotation_status") in ANNOTATION_STATUSES
        and record.get("reason_code") in REASON_CODES
        and isinstance(record.get("reason"), str)
        and bool(record["reason"].strip())
        and isinstance(objects, list)
        and isinstance(target_count, int)
        and not isinstance(target_count, bool)
        and target_count == len(objects)
        and (
            not config.get("enable_completeness_review", False)
            or record.get("review_performed") is True
        )
    )


async def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    image_dir = Path(config["image_dir"])
    checkpoint_path = Path(config["checkpoint_jsonl"])
    output_path = Path(config["output_json"])
    images = scan_images(image_dir, config["extensions"])
    latest = load_latest_records(checkpoint_path)
    successful = {
        relative_path
        for relative_path, record in latest.items()
        if is_reusable_success(record, config)
    }

    pending = [
        path
        for path in images
        if args.force or relative_image_path(path, image_dir) not in successful
    ]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit 不能小于 0")
        pending = pending[: args.limit]

    LOGGER.info("图片根目录: %s", image_dir)
    LOGGER.info("最终标注文件: %s", output_path)
    LOGGER.info("检查点文件: %s", checkpoint_path)
    LOGGER.info(
        "扫描到 %d 张图片，当前模型/提示词/生成参数已有 %d 张，本次计划 %d 张",
        len(images),
        len(successful),
        len(pending),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "image_root": str(image_dir),
                    "dataset_id": config.get("dataset_id"),
                    "annotation_file": str(output_path),
                    "checkpoint_file": str(checkpoint_path),
                    "model": config["model"],
                    "prompt_version": config["prompt_version"],
                    "generation_settings": generation_settings(config),
                    "discovered": len(images),
                    "already_successful": len(successful),
                    "scheduled": len(pending),
                    "sample_images": [
                        relative_image_path(path, image_dir) for path in pending[:10]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not pending:
        summary = build_summary(config, len(images), 0)
        await asyncio.to_thread(write_summary, summary, output_path)
        LOGGER.info("没有待处理图片，已刷新汇总文件: %s", output_path)
        return 0

    from industrial_fewshot.dependencies import load_openai_client
    AsyncOpenAI = load_openai_client()

    api_key_env = str(config.get("api_key_env", "VLLM_API_KEY"))
    api_key = os.getenv(api_key_env, str(config.get("api_key", "EMPTY")))
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config["base_url"],
        timeout=config["timeout_seconds"],
        max_retries=0,
    )
    writer = CheckpointWriter(checkpoint_path)
    prompt = build_prompt(config)
    response_format = build_response_format(config)
    queue: asyncio.Queue[Path | None] = asyncio.Queue()
    counters = {
        "scheduled": len(pending),
        "processed": 0,
        "success": 0,
        "failed": 0,
    }

    for image_path in pending:
        queue.put_nowait(image_path)
    worker_count = min(config["concurrency"], max(len(pending), 1))
    for _ in range(worker_count):
        queue.put_nowait(None)

    workers = [
        asyncio.create_task(
            worker(
                worker_id,
                queue,
                client,
                writer,
                image_dir,
                config,
                prompt,
                response_format,
                counters,
            )
        )
        for worker_id in range(1, worker_count + 1)
    ]

    try:
        await asyncio.gather(*workers)
    finally:
        await client.close()
        summary = build_summary(config, len(images), len(pending))
        await asyncio.to_thread(write_summary, summary, output_path)

    LOGGER.info(
        "任务完成：成功 %d，失败 %d；汇总已写入 %s",
        counters["success"],
        counters["failed"],
        output_path,
    )
    return 0 if counters["failed"] == 0 else 2


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        args = parse_args()
        config = load_config(args)
        return asyncio.run(run(args, config))
    except KeyboardInterrupt:
        LOGGER.warning("收到中断信号；已完成记录保留在检查点文件中")
        return 130
    except Exception as exc:
        LOGGER.error("程序退出: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
