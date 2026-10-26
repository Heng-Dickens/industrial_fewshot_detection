#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理历史遗留的空 LabelMe 标注文件。

早期版本的 diaogou_annotate.py 会把“确认没有目标”的图片也写出一个
``shapes`` 为空数组的 LabelMe JSON，导致标注文件数量远大于有效标注数量。
本脚本扫描图片目录，把 ``shapes`` 为空的同名 JSON 删除，只保留真正带框的标注。

安全约定：

- 只删除“图片旁边”的同名 JSON：同目录必须存在同名图片，否则一律不动。
  这样汇总文件 annotations.json、检查点 JSONL、参考示例标注等都不会被误删。
- JSON 无法解析、缺少 ``shapes`` 字段、``shapes`` 不是数组时一律不动。
- 默认直接删除；加 ``--dry-run`` 只列出将删除的文件，不产生任何修改。

用法：

    # 先预览，确认删除清单和数量
    python tool/clean_empty_annotations.py --dry-run

    # 正式删除（默认读取配置中的 image_dir）
    python tool/clean_empty_annotations.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    from industrial_fewshot.legacy.batch_annotate import read_json, resolve_path
except ImportError:  # 直接以脚本方式运行时回退到同目录导入
    from industrial_fewshot.legacy.batch_annotate import read_json, resolve_path

LOGGER = logging.getLogger("clean_empty_annotations")
SCRIPT_DIR = Path(__file__).resolve().parent
# 通用程序放在 tool/ 下，配置统一集中在上一层的 config/ 目录
from industrial_fewshot.paths import LEGACY_CONFIG_DIR
CONFIG_DIR = LEGACY_CONFIG_DIR
DEFAULT_CONFIG_NAME = "diaogou_annotation_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="删除 shapes 为空的 LabelMe JSON，只保留有标注的文件。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / DEFAULT_CONFIG_NAME,
        help=f"配置文件地址，默认 config/{DEFAULT_CONFIG_NAME}",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "要清理的图片目录；未指定时取配置中的 image_dir。"
            "指定 --root 后不再读取配置，可用 --extensions 单独控制扫描范围"
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        metavar="EXT",
        help="图片扩展名，仅在指定 --root 时生效；默认沿用配置或内置列表",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将被删除的文件和统计数量，不实际删除",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=20,
        help="最多打印多少个待删除文件路径，默认 20；设为 0 表示不打印清单",
    )
    return parser.parse_args()


def normalize_extensions(values: list[str] | tuple[str, ...]) -> set[str]:
    """把扩展名统一成带点的小写形式。"""
    normalized = set()
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        normalized.add(text if text.startswith(".") else f".{text}")
    return normalized


def load_target_dir(args: argparse.Namespace) -> tuple[Path, set[str]]:
    """确定要扫描的图片目录和扩展名集合。"""
    default_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.root is not None:
        image_dir = args.root.expanduser().resolve()
        extensions = (
            normalize_extensions(args.extensions)
            if args.extensions
            else default_extensions
        )
    else:
        config_path = args.config.expanduser().resolve()
        config = read_json(config_path)
        raw_image_dir = config.get("image_dir")
        if not raw_image_dir:
            raise ValueError(
                f"配置中没有 image_dir: {config_path}；也可以直接用 --root 指定目录"
            )
        image_dir = resolve_path(config_path.parent, raw_image_dir)
        raw_extensions = config.get("extensions")
        if isinstance(raw_extensions, list) and raw_extensions:
            extensions = normalize_extensions(raw_extensions)
        else:
            extensions = default_extensions
    if not extensions:
        raise ValueError("扩展名列表为空")
    if not image_dir.is_dir():
        raise ValueError(f"目录不存在: {image_dir}")
    return image_dir, extensions


def iter_annotation_paths(root: Path) -> Iterator[Path]:
    """递归产出目录下所有 .json 文件。"""
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def inspect_annotation(
    json_path: Path,
    image_dir: Path,
    image_extensions: set[str],
) -> tuple[str, str]:
    """检查一个 JSON，返回 (结论, 说明)。

    结论取值：

    - ``missing_image``：同目录没有同名图片，判定不是图片旁标注，不动
    - ``invalid``：JSON 无法解析或不是 LabelMe 结构，不动
    - ``empty``：是空标注，可以删除
    - ``kept``：有标注，保留
    """
    stem_path = json_path.with_suffix("")
    has_image = any(
        stem_path.with_suffix(extension).is_file()
        for extension in image_extensions
    )
    if not has_image:
        try:
            location = str(json_path.relative_to(image_dir))
        except ValueError:
            location = str(json_path)
        return "missing_image", f"同目录没有同名图片: {location}"

    try:
        document = read_json(json_path)
    except ValueError as exc:
        return "invalid", str(exc)
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        return "invalid", f"缺少 shapes 数组: {json_path}"

    if not shapes:
        return "empty", ""
    return "kept", ""


def delete_file(path: Path) -> None:
    """删除文件；Windows 上只读文件会先去掉只读属性。"""
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, 0o666)
        path.unlink()


def clean_directory(
    image_dir: Path,
    extensions: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    """扫描并清理一个目录，返回统计结果。"""
    counts = {
        "json_scanned": 0,
        "empty_deleted": 0,
        "kept": 0,
        "missing_image": 0,
        "invalid": 0,
        "failed": 0,
    }
    deleted: list[str] = []
    problems: list[str] = []

    for json_path in iter_annotation_paths(image_dir):
        counts["json_scanned"] += 1
        verdict, detail = inspect_annotation(json_path, image_dir, extensions)
        if verdict == "empty":
            if dry_run:
                counts["empty_deleted"] += 1
                deleted.append(str(json_path))
                continue
            try:
                delete_file(json_path)
            except OSError as exc:
                counts["failed"] += 1
                problems.append(f"删除失败 {json_path}: {exc}")
                continue
            counts["empty_deleted"] += 1
            deleted.append(str(json_path))
        elif verdict == "missing_image":
            counts["missing_image"] += 1
        elif verdict == "invalid":
            counts["invalid"] += 1
            problems.append(detail)
        else:
            counts["kept"] += 1

    return {"counts": counts, "deleted": deleted, "problems": problems}


def log_report(result: dict[str, Any], dry_run: bool, list_limit: int) -> None:
    counts = result["counts"]
    deleted = result["deleted"]
    problems = result["problems"]

    if list_limit > 0 and deleted:
        LOGGER.info(
            "%s以下文件%s：",
            "将删除" if dry_run else "已删除",
            f"（最多显示 {list_limit} 个）" if len(deleted) > list_limit else "",
        )
        for path in deleted[:list_limit]:
            LOGGER.info("  %s", path)
        if len(deleted) > list_limit:
            LOGGER.info("  ... 其余 %d 个略", len(deleted) - list_limit)

    if problems:
        LOGGER.warning("跳过或失败 %d 项，前若干条：", len(problems))
        for message in problems[:list_limit if list_limit > 0 else 10]:
            LOGGER.warning("  %s", message)

    LOGGER.info(
        "%s：扫描 %d 个 JSON，%s %d 个，保留有标注 %d 个，"
        "跳过无同名图片 %d 个，格式异常 %d 个，删除失败 %d 个",
        "预览结果" if dry_run else "清理完成",
        counts["json_scanned"],
        "待删除" if dry_run else "已删除",
        counts["empty_deleted"],
        counts["kept"],
        counts["missing_image"],
        counts["invalid"],
        counts["failed"],
    )
    if dry_run and counts["empty_deleted"]:
        LOGGER.info("这是预览模式，未删除任何文件；去掉 --dry-run 即可正式删除")


def run(args: argparse.Namespace) -> int:
    if args.list_limit < 0:
        raise ValueError("--list-limit 不能小于 0")
    image_dir, extensions = load_target_dir(args)
    LOGGER.info("扫描目录: %s", image_dir)
    LOGGER.info("图片扩展名: %s", " ".join(sorted(extensions)))

    result = clean_directory(image_dir, extensions, args.dry_run)
    log_report(result, args.dry_run, args.list_limit)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        LOGGER.warning("收到中断信号；已删除的文件不会恢复")
        return 130
    except Exception as exc:
        LOGGER.error("程序退出: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
