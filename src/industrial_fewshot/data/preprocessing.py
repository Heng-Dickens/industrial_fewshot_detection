"""Build reproducible query splits, COCO annotations and nested support manifests."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.config import load_config
from .dataset import inspect_dataset
from .labelme import load_evaluation_truth, write_json
from .supports import ReferencePool


def assign_splits(samples: list[dict], ratios: dict, seed: int, group_by: str) -> dict[str, list[dict]]:
    if (set(ratios) != {"train", "val", "test"} or any(type(v) not in (int, float)
            or not math.isfinite(v) or v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-8):
        raise ValueError("split_ratios 必须包含 train/val/test，比例非负且总和为 1")
    if group_by not in {"parent", "image"}:
        raise ValueError("group_by 只能是 parent 或 image")
    groups = {}
    for sample in samples:
        group = str(Path(sample["relative_path"]).parent) if group_by == "parent" else sample["relative_path"]
        groups.setdefault(group, []).append(sample)
    keys = sorted(groups, key=lambda key: (digest({"seed": seed, "group": key}), key))
    if len(keys) < sum(value > 0 for value in ratios.values()):
        raise ValueError("数据分组数量不足以形成指定的非空划分；不要把同一视频的相邻帧随机分散到测试集")
    names = ["train", "val", "test"]
    counts = {name: int(len(keys) * ratios[name]) for name in names}
    remainder_order = sorted(names, key=lambda name: (-(len(keys) * ratios[name] - counts[name]), names.index(name)))
    for name in remainder_order[:len(keys) - sum(counts.values())]:
        counts[name] += 1
    if any(ratios[name] > 0 and counts[name] == 0 for name in names):
        raise ValueError("当前分组数量过少，无法按比例得到非空 train/val/test")
    result, offset = {name: [] for name in names}, 0
    for name in names:
        result[name] = sorted([sample for key in keys[offset:offset + counts[name]] for sample in groups[key]],
                              key=lambda sample: sample["relative_path"])
        offset += counts[name]
    return result


def to_coco(samples: list[dict], labels: list[str], image_root: Path) -> dict:
    categories = [{"id": index, "name": label} for index, label in enumerate(labels, 1)]
    category_ids = {entry["name"]: entry["id"] for entry in categories}
    images, annotations = [], []
    for sample in samples:
        if sample["ground_truth"] is None:
            continue  # Unknown truth must not become a COCO negative implicitly.
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": sample["relative_path"], "width": sample["width"],
                       "height": sample["height"], "sha256": sample["image_sha256"]})
        for obj in sample["ground_truth"]:
            x1, y1, x2, y2 = obj["bbox_pixels_xyxy"]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                "category_id": category_ids[obj["label"]], "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    return {"info": {"image_root": str(image_root), "unknown_truth_excluded": sum(s["ground_truth"] is None for s in samples)},
            "images": images, "annotations": annotations, "categories": categories}


def prepare_experiment_dataset(config: dict, *, dry_run=False) -> dict:
    settings = config["dataset_preparation"]
    required = {"prepared_dir", "splits_dir", "supports_dir"}
    if not required.issubset(settings):
        raise ValueError("preparation 必须配置 prepared_dir/splits_dir/supports_dir")
    counts = settings.get("support_counts", [1, 5, 10, 25])
    seeds = settings.get("support_seeds", [0])
    if (not isinstance(counts, list) or not counts or any(type(k) is not int or k < 1 for k in counts)
            or len(set(counts)) != len(counts) or not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("support_counts 必须为不重复正整数，support_seeds 必须为不重复整数")
    inspect_config = {**config, "mode": "test", "dataset_manifest": None}
    items = inspect_dataset(inspect_config)
    target = config["targets"][0]
    labels = [target["label"]]
    pools = {seed: ReferencePool(target["config"], seed) for seed in seeds}
    pool = pools[seeds[0]]
    if max(counts) > len(pool.positives):
        raise ValueError(f"支持集正样本不足: 最大档位 {max(counts)}，实际 {len(pool.positives)}")
    samples, excluded_overlap, excluded_duplicates, seen = [], [], [], {}
    for item in items:
        checksum = image_sha256(item["path"])
        if checksum in pool.original_hashes:
            excluded_overlap.append(item["relative_path"])
            continue
        truth_path = config["ground_truth_dir"] / Path(item["relative_path"]).with_suffix(".json")
        truth, provenance = load_evaluation_truth(truth_path, item["width"], item["height"], set(labels), config["missing_ground_truth"])
        sample = {key: item[key] for key in ("relative_path", "width", "height")}
        sample.update(image_sha256=checksum, ground_truth=truth, ground_truth_source=provenance,
                      annotation_sha256=image_sha256(truth_path) if truth_path.is_file() else None)
        if checksum in seen:
            if seen[checksum]["ground_truth"] != truth:
                raise ValueError(f"重复图片的真值冲突: {item['relative_path']}")
            excluded_duplicates.append(item["relative_path"])
            continue
        seen[checksum] = sample
        samples.append(sample)
    if not samples:
        raise ValueError("去除参考重叠和重复图片后没有 query 图片")
    split_seed = settings.get("split_seed", 0)
    if type(split_seed) is not int:
        raise ValueError("split_seed 必须是整数")
    ratios = settings.get("split_ratios", {"train": 0, "val": 0, "test": 1})
    splits = assign_splits(samples, ratios, split_seed, settings.get("group_by", "parent"))
    snapshot = digest({"images": [{key: item[key] for key in ("relative_path", "image_sha256", "ground_truth")} for item in samples]})
    prepared, splits_dir, supports_dir = (Path(settings[key]).resolve() for key in ("prepared_dir", "splits_dir", "supports_dir"))
    for kind, output in (('prepared', prepared), ('splits', splits_dir), ('supports', supports_dir)):
        for protected in (config["image_dir"], config["ground_truth_dir"], pool.root):
            if output == protected or output.is_relative_to(protected):
                # Unified references keep only shot manifests and preparation
                # metadata alongside the protected positive/negative assets.
                if (protected == pool.root and protected not in (config['image_dir'], config['ground_truth_dir'])
                        and ((kind == 'supports' and output == pool.root)
                             or (kind == 'prepared' and (output == pool.root / 'preparation'
                                 or output.is_relative_to(pool.root / 'preparation'))))):
                    continue
                raise ValueError(f"派生数据输出不能写入原图/真值/参考池内: {output}")
    report = {"schema": "industrial_fewshot.prepared.v1", "image_root": str(config["image_dir"]),
              "ground_truth_root": str(config["ground_truth_dir"]), "image_count": len(samples),
              "split_counts": {name: len(rows) for name, rows in splits.items()}, "dataset_sha256": snapshot,
              "excluded_reference_overlap": excluded_overlap, "excluded_duplicate_images": excluded_duplicates,
              "support_counts": counts, "support_seeds": seeds, "positive_pool_count": len(pool.positives),
              "negative_pool_count": len(pool.negatives), "dry_run": dry_run}
    if dry_run:
        return report
    write_json(prepared / "annotations.json", to_coco(samples, labels, config["image_dir"]))
    for name, rows in splits.items():
        write_json(splits_dir / (name + ".json"), {"schema": "industrial_fewshot.split.v1", "split": name,
                   "image_root": str(config["image_dir"]), "ground_truth_root": str(config["ground_truth_dir"]),
                   "dataset_sha256": snapshot, "samples": rows, "split_seed": split_seed,
                   "group_by": settings.get("group_by", "parent"), "split_ratios": ratios})
    for seed, selected_pool in pools.items():
        def entry(row):
            return {"sample_id": row["sample_id"], "image_path": str(selected_pool.path(row["image"])),
                    "image_sha256": image_sha256(selected_pool.path(row["image"])),
                    "json_path": str(selected_pool.path(row["annotation"])) if row.get("annotation") else None,
                    "annotation_sha256": image_sha256(selected_pool.path(row["annotation"])) if row.get("annotation") else None,
                    "boxed_image": str(selected_pool.path(row["boxed_image"])) if row.get("boxed_image") else None,
                    "crops": [str(selected_pool.path(c["image"])) for c in row.get("crops", [])]}
        for count in counts:
            positives, negatives = selected_pool.select(count, min(count, len(selected_pool.negatives)))
            pinned = selected_pool.pinned_manifest(count) or {}
            write_json(supports_dir / f"{count}shot/seed_{seed}.json", {
                "schema": "industrial_fewshot.support.v1", "positive_count": count, "selection_seed": seed,
                "reference_root": str(selected_pool.root), "positive_samples": [entry(row) for row in positives],
                "negative_samples": [entry(row) for row in negatives],
                **{key: pinned[key] for key in ('selection_policy', 'selection_provenance', 'assets_root') if key in pinned},
                "query_dataset_sha256": snapshot})
    write_json(prepared / "preparation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_experiment_dataset(load_config(args.config), dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, str(exc) + "\n")
