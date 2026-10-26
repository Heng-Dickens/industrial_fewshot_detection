"""Draw LabelMe annotation boxes onto images for visual review.

Inputs are an image directory and a directory of LabelMe-style prediction
JSONs (one per image, as produced under a run's predictions/). Output images
with drawn boxes go to a new directory, by default under data/visualized/.
Images without an annotation JSON are copied unchanged so the output stays a
complete review set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOX_COLOR = (255, 48, 48)
TEXT_COLOR = (255, 255, 255)
BOX_WIDTH = 3


def load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_document(image_path: Path, document: dict, output_path: Path, font: ImageFont.ImageFont) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for shape in document.get("shapes", []):
        points = shape.get("points", [])
        if shape.get("shape_type") not in {"rectangle", "polygon"} or len(points) < 2:
            continue
        label = str(shape.get("label", ""))
        confidence = shape.get("confidence")
        caption = label if confidence is None else f"{label} {confidence:.2f}"
        polygon = [(float(x), float(y)) for x, y in points]
        if shape["shape_type"] == "rectangle":
            draw.rectangle(polygon, outline=BOX_COLOR, width=BOX_WIDTH)
            anchor = polygon[0]
        else:
            draw.polygon(polygon, outline=BOX_COLOR)
            anchor = min(polygon, key=lambda p: (p[1], p[0]))
        text_origin = (anchor[0], anchor[1] - font.size - 2)
        if text_origin[1] < 0:
            text_origin = (anchor[0], anchor[1] + 2)
        box = draw.textbbox(text_origin, caption, font=font)
        draw.rectangle(box, fill=BOX_COLOR)
        draw.text(text_origin, caption, fill=TEXT_COLOR, font=font)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, help="Directory holding the original images")
    parser.add_argument("--annotation-dir", required=True, help="Directory holding LabelMe prediction JSONs")
    parser.add_argument("--output-dir", default=None,
                        help="Destination (default: data/visualized/<annotation dir name>)")
    parser.add_argument("--label", action="append", default=None,
                        help="Only draw shapes with this label (repeatable; default: all)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "visualized" / (
        annotation_dir.parent.name if annotation_dir.name == "predictions" else annotation_dir.name)
    if not output_dir.resolve().is_relative_to(PROJECT_ROOT):
        raise SystemExit("Output must stay inside this project under data/.")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()

    images = {p.name: p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    jsons = sorted(annotation_dir.glob("*.json"))
    if not jsons:
        raise SystemExit(f"No annotation JSONs under {annotation_dir}")

    drawn, copied, missing = 0, 0, []
    for json_path in jsons:
        document = json.loads(json_path.read_text(encoding="utf-8-sig"))
        image_name = Path(document.get("imagePath", json_path.stem + ".jpg")).name
        image_path = images.get(image_name)
        if image_path is None:
            missing.append(image_name)
            continue
        shapes = document.get("shapes", [])
        if args.label:
            shapes = [s for s in shapes if s.get("label") in args.label]
            document = {**document, "shapes": shapes}
        draw_document(image_path, document, output_dir / image_path.name, font)
        drawn += 1
        images.pop(image_name)

    for image_path in sorted(images.values()):
        with Image.open(image_path) as source:
            source.convert("RGB").save(output_dir / image_path.name)
        copied += 1

    print(f"drawn: {drawn}, copied without boxes: {copied}, missing images: {len(missing)}")
    print(f"output: {output_dir}")
    if missing:
        print("missing image files:", ", ".join(missing))


if __name__ == "__main__":
    main()
