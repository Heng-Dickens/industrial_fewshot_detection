"""Shared dataset scanning and local-only evaluation truth."""
from pathlib import Path
from PIL import Image
from .labelme import load_evaluation_truth, read_json
from industrial_fewshot.labels import all_label_names

def inspect_dataset(config: dict, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("--limit 必须大于 0")
    root = config["image_dir"]
    references = [Path(t["config"]["reference_image_dir"]) for t in config["targets"] if t["config"].get("reference_image_dir")]
    paths = sorted(p for p in (root.rglob("*") if config["recursive"] else root.glob("*"))
                   if p.is_file() and p.suffix.lower() in config["extensions"]
                   and not any(p.resolve().is_relative_to(ref) for ref in references))
    frozen = {}
    if config.get("dataset_manifest"):
        manifest = read_json(Path(config["dataset_manifest"]))
        if manifest.get("schema") != "industrial_fewshot.split.v1" or Path(manifest["image_root"]).resolve() != root.resolve():
            raise ValueError("数据划分清单的格式或图片根目录不匹配")
        frozen = {row["relative_path"]: row for row in manifest["samples"]}
        if len(frozen) != len(manifest["samples"]):
            raise ValueError("划分清单包含重复图片")
        paths = []
        for relative in sorted(frozen):
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"划分清单图片缺失或越界: {relative}")
            paths.append(path)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError("没有待标注图片（参考目录中的图片自动排除）")
    items, destinations = [], set()
    labels = set(all_label_names(config['targets']))
    for image in paths:
        relative = image.relative_to(root)
        key = str(relative.with_suffix(".json")).casefold()
        if key in destinations:
            raise ValueError(f"不同图片映射到同一标注文件，请重命名: {relative}")
        destinations.add(key)
        with Image.open(image) as source:
            width, height = source.size
        truth_path = config["ground_truth_dir"] / relative.with_suffix(".json")
        truth = None
        truth_source = "not_evaluated"
        if config["mode"] == "train":
            truth, truth_source = load_evaluation_truth(truth_path, width, height, labels, config["missing_ground_truth"])
        if relative.as_posix() in frozen:
            expected = frozen[relative.as_posix()]
            if width != expected["width"] or height != expected["height"]:
                raise ValueError("图片尺寸与固定划分不一致，请重新准备数据")
            if config["mode"] == "train" and truth != expected["ground_truth"]:
                raise ValueError("真值与固定划分不一致，请重新准备数据；已成功推理仍可复用")
        items.append({"path": image, "relative_path": relative.as_posix(), "width": width, "height": height,
                      "expected_image_sha256": frozen.get(relative.as_posix(), {}).get("image_sha256"),
                      "ground_truth": truth, "ground_truth_source": truth_source})
    if config["mode"] == "train" and all(i["ground_truth"] is None for i in items):
        raise ValueError("train 模式没有可评估图片：请提供 LabelMe 真值，或为已确认的负样本设置 missing_ground_truth=negative")
    return items

