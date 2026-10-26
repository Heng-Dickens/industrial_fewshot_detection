"""Paired positive materials and nested reference counts for controlled experiments."""
from __future__ import annotations

import hashlib
import copy
from pathlib import Path

from industrial_fewshot.data.labelme import read_json
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.prompts.references import compact_labelme_document, encode_data_url, render_reference_image
from industrial_fewshot.paths import stable_reference_identity, resolve_location

DEFAULT_VARIANTS = ["O+J", "O+C", "O+B", "O+C+J", "O+B+J", "O+C+B", "O+C+B+J"]
MATERIALS = {"O": "original", "C": "crop", "B": "boxed", "J": "json"}


def reference_pool(config: dict, seed: int):
    return ClassReferencePool(config, seed) if config.get('reference_per_class') else ReferencePool(config, seed)


class ClassReferencePool:
    """Compose k independent supports per class into one multi-class image request."""

    def __init__(self, config: dict, seed: int):
        self.pools = []
        root = Path(config['reference_image_dir']).resolve()
        for label in config['labels']:
            directory = (root / label['name']).resolve()
            if not directory.is_relative_to(root):
                raise ValueError('Reference class directory escapes reference root')
            settings = {**config, 'labels': [label], 'reference_image_dir': str(directory),
                        'support_manifest_dir': str(Path(config['support_manifest_dir']) / label['name'])}
            self.pools.append((label['name'], ReferencePool(settings, seed)))
        self.original_hashes = set().union(*(pool.original_hashes for _, pool in self.pools))
        if len(self.original_hashes) != sum(len(pool.original_hashes) for _, pool in self.pools):
            raise ValueError('Different classes reuse the same reference image content')

    def prepare(self, config: dict, spec: dict) -> dict:
        keys = ('reference_samples', 'reference_originals', 'reference_examples', 'reference_crops',
                'reference_boxed_images', 'reference_negatives')
        combined = {key: [] for key in keys}
        selections, fingerprints = {}, []
        for label, pool in self.pools:
            temporary = copy.copy(config)
            selections[label] = pool.prepare(temporary, spec)
            for row in temporary['reference_samples']:
                row['label'] = label
            for key in keys:
                combined[key].extend(temporary[key])
            fingerprints.append({'label': label, 'sha256': temporary['reference_manifest_sha256']})
        config.update(combined, reference_mode='composed',
                      reference_manifest_sha256=digest({'schema': 'class_references.v1', 'sets': fingerprints}))
        return {'positive_sample_ids': [row['sample_id'] for row in combined['reference_samples']],
                'negative_sample_ids': [row['sample_id'] for row in combined['reference_negatives']],
                'positive_count_per_class': spec['positive_count'], 'class_selections': selections,
                'source_reference_count': sum(r['source_reference_count'] for r in selections.values()),
                'reference_image_count': sum(r['reference_image_count'] for r in selections.values())}


def suite_settings(raw: dict | None) -> dict:
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("experiment_suite 必须是配置对象")
    settings = {"enabled": False, "include_prompt_only": True, "positive_counts": [5],
                "negative_variants": [False, True], "positive_variants": DEFAULT_VARIANTS,
                "selection_seed": 0, "exclude_reference_overlap": True, **raw}
    for key in ("enabled", "include_prompt_only", "exclude_reference_overlap"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"experiment_suite.{key} 必须是布尔值")
    counts = settings["positive_counts"]
    if (not isinstance(counts, list) or not counts or any(type(k) is not int or k < 1 for k in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError("experiment_suite.positive_counts 必须是不重复的正整数列表")
    negatives = settings["negative_variants"]
    if (not isinstance(negatives, list) or not negatives or any(type(n) is not bool for n in negatives)
            or len(set(negatives)) != len(negatives)):
        raise ValueError("experiment_suite.negative_variants 必须是不重复的布尔列表")
    if type(settings["selection_seed"]) is not int:
        raise ValueError("experiment_suite.selection_seed 必须是整数")
    variants = settings["positive_variants"]
    if (not isinstance(variants, list) or not variants or any(v not in DEFAULT_VARIANTS for v in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError(f"experiment_suite.positive_variants 可选 {DEFAULT_VARIANTS}，不包含仅 O")
    return settings


def group_specs(settings: dict) -> list[dict]:
    groups = []
    if settings["include_prompt_only"]:
        groups.append({"id": "prompt_only", "variant": "text_only", "positive_count": 0,
                       "negative_count": 0, "positive_inputs": []})
    for count in settings["positive_counts"]:
        for variant in settings["positive_variants"]:
            for with_negative in settings["negative_variants"]:
                groups.append({"id": f"{variant.replace('+', '_')}_k{count:02d}_" + ("posneg" if with_negative else "pos"),
                               "variant": variant, "positive_count": count,
                               "negative_count": count if with_negative else 0,
                               "positive_inputs": [MATERIALS[k] for k in variant.split("+")]})
    return groups


class ReferencePool:
    """Read prepared pairs once, freeze assets, and select stable nested prefixes."""

    def __init__(self, config: dict, seed: int):
        self.config, self.seed = config, seed
        if not config.get("reference_image_dir"):
            raise ValueError("experiment_suite 需要 reference_image_dir 指向整理后的数据目录")
        self.root = Path(config["reference_image_dir"]).resolve()
        manifest = read_json(self.root / "manifest.json")
        if manifest.get("schema") != "annotationtool.reference_dataset.v1":
            raise ValueError("experiment_suite 需要 tool/prepare_reference_data.py 整理的数据及 manifest.json")
        self.positives, self.negatives, self.original_hashes = [], [], set()
        self.assets, self.documents = {}, {}
        labels = {label["name"] for label in config["labels"]}
        seen_ids = set()
        for source in manifest["images"]:
            row = dict(source)
            row["sample_id"] = row["image"]
            if row["sample_id"] in seen_ids:
                raise ValueError(f"参考清单包含重复图像: {row['image']}")
            seen_ids.add(row["sample_id"])
            if row["split"] == "positive":
                document = read_json(self.path(row["annotation"]))
                if not any(s.get("label") in labels for s in document.get("shapes", [])):
                    continue
                self.documents[row["sample_id"]] = document
                self.positives.append(row)
            elif row["split"] == "negative":
                if row.get("annotation") and read_json(self.path(row["annotation"])).get("shapes"):
                    raise ValueError(f"负样本包含非空标注: {row['image']}")
                self.negatives.append(row)
            else:
                raise ValueError(f"未知参考样本 split: {row['split']}")
            checksum = image_sha256(self.path(row["image"]))
            if checksum in self.original_hashes:
                raise ValueError(f"参考原图内容重复，不能作为多个独立样本计数: {row['image']}")
            self.original_hashes.add(checksum)
        order = lambda r: (digest({"seed": seed, "sample_id": r["sample_id"]}), r["sample_id"])
        self.positives.sort(key=order)
        self.negatives.sort(key=order)

    def pinned_manifest(self, count: int) -> dict | None:
        directory = self.config.get('support_manifest_dir')
        if not directory:
            return None
        path = Path(directory) / f'{count}shot/seed_{self.seed}.json'
        if not path.is_file():
            return None
        document = read_json(path)
        if document.get('selection_policy') != 'explicit':
            return None
        if (document.get('schema') != 'industrial_fewshot.support.v1'
                or document.get('positive_count') != count or document.get('selection_seed') != self.seed
                or resolve_location(Path(document['reference_root'])) != self.root):
            raise ValueError(f'Explicit support manifest settings do not match: {path}')
        return document

    def select(self, count: int, negative_count: int = 0) -> tuple[list[dict], list[dict]]:
        """Honor explicitly pinned supports; unpinned sets retain seeded ordering."""
        if count > len(self.positives) or negative_count > len(self.negatives):
            raise ValueError(f'参考数量不足: 正样本 {count}、负样本 {negative_count}')
        pinned = self.pinned_manifest(count)
        if pinned is None:
            return self.positives[:count], self.negatives[:negative_count]
        selected = []
        for key, pool, required in (('positive_samples', self.positives, count),
                                    ('negative_samples', self.negatives, negative_count)):
            rows = pinned.get(key, [])
            ids = [row['sample_id'] for row in rows]
            available = {row['sample_id']: row for row in pool}
            if (len(ids) != count or len(set(ids)) != len(ids) or required > len(ids)
                    or any(sample_id not in available for sample_id in ids)):
                raise ValueError(f'Explicit support selection is invalid: {key}')
            selected.append([available[sample_id] for sample_id in ids[:required]])
        return tuple(selected)

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"参考资源缺失或位于参考目录之外: {relative}")
        return path

    def asset(self, relative: str) -> dict:
        if relative not in self.assets:
            path = self.path(relative)
            rendered, mime = render_reference_image(path, [])
            self.assets[relative] = {"image_path": str(path), "image_sha256": image_sha256(path),
                                     "rendered_sha256": hashlib.sha256(rendered).hexdigest(),
                                     "rendered_data_url": encode_data_url(rendered, mime)}
        return self.assets[relative]

    def prepare(self, config: dict, spec: dict) -> dict:
        k, n = spec["positive_count"], spec["negative_count"]
        if k > len(self.positives) or n > len(self.negatives):
            raise ValueError(f"参考数量不足: 要求正样本 {k}、负样本 {n}；"
                             f"实际正样本 {len(self.positives)}、负样本 {len(self.negatives)}，不会静默截断")
        materials = spec["positive_inputs"]
        positive_rows, negative_rows = self.select(k, n)
        samples = []
        for row in positive_rows:
            sample = {"sample_id": row["sample_id"], "original": self.asset(row["image"])}
            if "crop" in materials:
                if not row.get("crops"):
                    raise ValueError(f"正样本缺少裁剪图: {row['image']}")
                sample["crops"] = [self.asset(c["image"]) for c in row["crops"]]
            if "boxed" in materials:
                if not row.get("boxed_image"):
                    raise ValueError(f"正样本缺少标注框可见图: {row['image']}")
                sample["boxed"] = self.asset(row["boxed_image"])
            if "json" in materials:
                sample["json"] = {"json_path": str(self.path(row["annotation"])),
                                  "image_path": sample["original"]["image_path"],
                                  "compact_json": compact_labelme_document(self.documents[row["sample_id"]])}
            samples.append(sample)
        negatives = [dict(self.asset(row["image"]), sample_id=row["sample_id"]) for row in negative_rows]
        config.update(reference_mode="composed", reference_samples=samples,
                      reference_originals=[s["original"] for s in samples],
                      reference_examples=[s["json"] for s in samples if "json" in s],
                      reference_crops=[c for s in samples for c in s.get("crops", [])],
                      reference_boxed_images=[s["boxed"] for s in samples if "boxed" in s],
                      reference_negatives=negatives)
        # Fingerprint only material actually sent. A change in unused JSON/boxed/crop
        # files must not cause unrelated groups to consume tokens again.
        def fingerprint(value):
            if isinstance(value, list):
                return [fingerprint(v) for v in value]
            if isinstance(value, dict):
                return {key: fingerprint(v) for key, v in value.items() if key != "rendered_data_url"}
            return value
        config["reference_manifest_sha256"] = digest(stable_reference_identity({"schema": "paired_references.v1",
            "samples": fingerprint(samples), "negatives": fingerprint(negatives)}))
        return {"positive_sample_ids": [s["sample_id"] for s in samples],
                "negative_sample_ids": [s["sample_id"] for s in negatives],
                "source_reference_count": k + n,
                "reference_image_count": k + n + len(config["reference_crops"]) + len(config["reference_boxed_images"])}


