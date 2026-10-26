"""Verify that each detector uses the frozen, nested support selection."""
from pathlib import Path

from .labelme import read_json
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.paths import resolve_location


def validate_support_selection(config: dict) -> list[dict]:
    directory = config.get("support_manifest_dir")
    if not directory:
        return []
    verified = []
    for target in config["targets"]:
        settings = target["config"]
        if settings.get('reference_per_class') and settings.get('reference_mode') == 'composed':
            for label in settings['labels']:
                name = label['name']
                samples = [row for row in settings['reference_samples'] if row.get('label') == name]
                child = {**settings, 'reference_per_class': False, 'reference_samples': samples}
                reports = validate_support_selection({**config,
                    'support_manifest_dir': str(Path(directory) / name),
                    'targets': [{**target, 'id': f"{target['id']}/{name}", 'config': child}]})
                verified.extend(reports)
            continue
        if target["backend"] in {"yoloe", "yoloe_fusion", "pet_dino"}:
            positives = settings["reference_examples"]
            seed = settings[target['backend']]["selection_seed"]
        elif settings.get("reference_mode") == "composed":
            positives = settings.get("reference_samples", [])
            seed = config["experiment_suite"]["selection_seed"]
        else:
            continue
        if not positives:
            continue
        path = Path(directory) / f"{len(positives)}shot/seed_{seed}.json"
        frozen = read_json(path)
        if (frozen.get("schema") != "industrial_fewshot.support.v1"
                or frozen.get("positive_count") != len(positives) or frozen.get("selection_seed") != seed):
            raise ValueError(f"Support manifest settings do not match: {path}")
        negatives = settings.get("reference_negatives", [])
        for name, actual in (("positive_samples", positives), ("negative_samples", negatives)):
            expected = frozen[name][:len(actual)]
            if [r["sample_id"] for r in actual] != [r["sample_id"] for r in expected]:
                raise ValueError(f"Support selection differs from frozen manifest: {path} / {name}")
            for row in expected:
                if image_sha256(resolve_location(Path(row["image_path"]))) != row["image_sha256"]:
                    raise ValueError(f"Support image changed; prepare the dataset again: {row['sample_id']}")
                if row.get("annotation_sha256") and image_sha256(resolve_location(Path(row["json_path"]))) != row["annotation_sha256"]:
                    raise ValueError(f"Support annotation changed; prepare the dataset again: {row['sample_id']}")
        verified.append({"target_id": target["id"], "path": str(path), "sha256": image_sha256(path),
                         "positive_count": len(positives), "negative_count": len(negatives)})
    return verified
