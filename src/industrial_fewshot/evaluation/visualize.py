"""Render saved predictions with local GT; never call a detector."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .compare import load_saved_group
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.data.labelme import load_evaluation_truth, write_json
from industrial_fewshot.postprocess.schema import validate_prediction
from industrial_fewshot.data.labelme import validate_pixel_prediction
from industrial_fewshot.paths import resolve_location
from .detection_metrics import evaluation_settings


def render_run(run_dir: Path, output: Path, *, score_threshold: float | None = None) -> dict:
    group = load_saved_group(run_dir)
    settings = dict(group.get("evaluation") or {})
    if score_threshold is not None:
        settings["score_threshold"] = score_threshold
    score_threshold = evaluation_settings(settings)["score_threshold"]
    source, truth_root, output = resolve_location(Path(group["image_dir"])), resolve_location(Path(group["ground_truth_dir"])), output.resolve()
    for protected in (source, truth_root, Path(group["run_dir"])):
        if output == protected.resolve() or output.is_relative_to(protected.resolve()):
            raise ValueError("Visualization output must be outside originals, GT and saved run")
    targets = {target["id"]: target for target in group["targets"]}
    rows = []
    for record in group["saved_records"]:
        path = source / record["relative_path"]
        if image_sha256(path) != record.get("image_sha256"):
            raise ValueError(f"Query image differs from saved record: {path}")
        with Image.open(path) as original:
            image = original.convert("RGB")
        width, height = image.size
        target = targets[record["target_id"]]
        truth, _ = load_evaluation_truth(truth_root / Path(record["relative_path"]).with_suffix(".json"),
                                       width, height, {target["label"]}, group["missing_ground_truth"])
        draw = ImageDraw.Draw(image)
        for obj in truth or []:
            draw.rectangle(obj["bbox_pixels_xyxy"], outline="lime", width=3)
        candidate_count = prediction_count = 0
        if record["status"] == "success":
            validator = validate_pixel_prediction if target.get("backend") in {"yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"} else validate_prediction
            validated = validator(record, {target["label"]}, width, height)
            candidate_count = len(validated["objects"])
            for obj in validated["objects"]:
                if obj["confidence"] < score_threshold:
                    continue
                draw.rectangle(obj["bbox_pixels_xyxy"], outline="red", width=3)
                prediction_count += 1
        else:
            draw.text((12, 12), "INFERENCE FAILED", fill="red")
        destination = output / target["id"] / Path(record["relative_path"]).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        rows.append({"relative_path": record["relative_path"], "status": record["status"], "output": str(destination),
                     "candidate_count": candidate_count, "prediction_count": prediction_count,
                     "filtered_count": candidate_count - prediction_count})
    report = {"run_dir": group["run_dir"], "score_threshold": score_threshold,
              "legend": {"ground_truth": "green", "prediction": "red"}, "images": rows}
    write_json(output / "manifest.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, help="筛选显示框的分数阈值；默认使用运行时保存的阈值")
    args = parser.parse_args(argv)
    report = render_run(args.run_dir, args.output_dir, score_threshold=args.score_threshold)
    print(f"Rendered {len(report['images'])} images to {args.output_dir}")
    return 0
