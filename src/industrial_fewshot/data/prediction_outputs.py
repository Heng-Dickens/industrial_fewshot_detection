"""Export parallel score-filtered labels without discarding raw predictions."""
import os
from pathlib import Path

from .labelme import build_labelme_document, write_json
from industrial_fewshot.evaluation.detection_metrics import output_score_thresholds


def export_prediction_thresholds(target_dir: Path, item: dict, record: dict, settings: dict,
                                *, include_primary: bool = True) -> dict:
    primary = settings["score_threshold"]
    paths = {}
    for threshold in output_score_thresholds(settings):
        folder = "predictions" if threshold == primary else f"predictions_score_{threshold:g}"
        destination = target_dir / folder / Path(item["relative_path"]).with_suffix(".json")
        paths[f"{threshold:g}"] = str(destination)
        if threshold == primary and not include_primary:
            continue
        objects = [obj for obj in record["objects"] if obj["confidence"] >= threshold]
        write_json(destination, build_labelme_document(
            os.path.relpath(item["path"], destination.parent), item["width"], item["height"], objects))
    return {"labelme_json": paths[f"{primary:g}"], "labelme_json_by_score": paths}
