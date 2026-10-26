"""Safe YAML/JSON composition; each relative path belongs to its declaring file."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .paths import resolve_location

PATH_KEYS = {"image_dir", "ground_truth_dir", "reference_image_dir", "prompt_file", "review_prompt_file",
             "output_dir", "output_json", "checkpoint_jsonl", "source_dir", "weights", "checkpoint",
             "manifest_file", "support_manifest_dir", "prepared_dir", "splits_dir", "supports_dir", "coco_file", "api_key_file", "env_file", "text_encoder_weights", "resume_comparison_dir", "worker_python", "model_config", "language_model_dir"}


def redact(value):
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, dict):
        return {key: "<redacted>" if key.lower() in {"api_key", "token", "password", "authorization", "access_token", "secret"}
                else redact(child) for key, child in value.items()}
    return value


def redact(value):
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, dict):
        return {key: "<redacted>" if key.lower() in {"api_key", "token", "password", "authorization", "access_token", "secret"}
                else redact(child) for key, child in value.items()}
    return value


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def anchor_paths(value, base: Path):
    if isinstance(value, list):
        return [anchor_paths(v, base) for v in value]
    if not isinstance(value, dict):
        return value
    return {key: str(resolve_location(base / child)) if key in PATH_KEYS and isinstance(child, str) and child.strip()
            else anchor_paths(child, base) for key, child in value.items()}


def load_document(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict, list[str]]:
    path = resolve_location(path)
    if path in stack or len(stack) > 32:
        raise ValueError(f"配置 extends 循环或嵌套过深: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("读取 YAML 配置需要 PyYAML，请安装项目基础依赖") from exc
        raw = yaml.safe_load(text)
    else:
        raise ValueError(f"配置必须是 YAML 或 JSON: {path}")
    if not isinstance(raw, dict):
        raise ValueError(f"配置顶层必须是对象: {path}")
    parents = raw.pop("extends", [])
    parents = [parents] if isinstance(parents, str) else parents
    if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
        raise ValueError("extends 必须是配置路径或路径列表")
    merged, sources = {}, []
    for parent in parents:
        inherited, files = load_document(path.parent / parent, (*stack, path))
        merged = deep_merge(merged, inherited)
        sources.extend(files)
    sources.append(str(path))
    return deep_merge(merged, anchor_paths(raw, path.parent)), list(dict.fromkeys(sources))
