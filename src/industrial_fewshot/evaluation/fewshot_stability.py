"""Aggregate independent support seeds from verified comparison reports."""
import argparse
import json
from pathlib import Path
import statistics

from industrial_fewshot.cache import digest
from industrial_fewshot.data.labelme import read_json, write_json

METRICS = ("mAP50", "mAP50_95", "recall", "precision", "image_accuracy", "negative_error_rate", "proxy_loss")


def summarize(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("No comparison reports")
    cohort = digest({key: reports[0][key] for key in ("dataset_sha256", "evaluation", "missing_ground_truth")})
    grouped = {}
    for report in reports:
        if digest({key: report[key] for key in ("dataset_sha256", "evaluation", "missing_ground_truth")}) != cohort:
            raise ValueError("Stability analysis requires the same query cohort, GT and evaluation settings")
        for row in report["rows"]:
            key = tuple(row.get(k) for k in ("backend", "model", "reference_mode", "positive_original_count", "negative_original_count"))
            seed = row.get("selection_seed")
            if seed is None:
                raise ValueError("Run metadata has no selection_seed; cannot establish independent support selections")
            group = grouped.setdefault(key, {})
            if seed in group:
                raise ValueError("Duplicate support seed within one method; repeated cached runs are not independent samples")
            group[seed] = row
    rows = []
    for key, samples in grouped.items():
        row = dict(zip(("backend", "model", "reference_mode", "positive_original_count", "negative_original_count"), key))
        row.update(seed_count=len(samples), seeds=sorted(samples))
        for metric in METRICS:
            values = [sample[metric] for sample in samples.values() if sample.get(metric) is not None]
            row[metric] = {"count": len(values), "mean": statistics.mean(values) if values else None,
                           "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        row["failed_request_count"] = sum(sample["failed_request_count"] for sample in samples.values())
        rows.append(row)
    return {"dataset_sha256": reports[0]["dataset_sha256"], "rows": rows,
            "policy": "Sample standard deviation across distinct support seeds; undefined for one seed. Failed requests remain in source metrics."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize([read_json(path) for path in args.comparison])
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
