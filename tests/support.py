"""Shared fixtures and fake clients; no network access."""
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from industrial_fewshot.data.labelme import write_json

LABEL = "hanging object"


def response(label="target", confidence=0.9):
    obj = {"label": label, "bbox_2d": [100, 200, 800, 900], "hanging_object_confirmed": True}
    if confidence is not None:
        obj["confidence"] = confidence
    return json.dumps({"annotation_status": "annotated", "reason_code": "target_detected", "reason": "目标清晰可见", "target_count": 1, "objects": [obj]})



class FakeClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []
        self.closed = False
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        payload = next(self.payloads)
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload), finish_reason="stop")], usage=None)

    async def close(self):
        self.closed = True



def fixture(root, mode="train", backend="generic"):
    images = root / "images"
    images.mkdir()
    refs = root / "references"
    refs.mkdir()
    Image.new("RGB", (1000, 500), "gray").save(images / "frame.jpg")
    truth = {"imageWidth": 1000, "imageHeight": 500, "shapes": [{"label": "target", "shape_type": "rectangle", "points": [[100, 100], [800, 450]]}]}
    write_json(images / "frame.json", truth)
    (root / "prompt.txt").write_text("CUSTOM_EXTERNAL_RULE {{label_name}} {{label_description}} {{label_lines}}", encoding="utf-8")
    (root / "review.txt").write_text("CUSTOM_REVIEW {{initial_result}}", encoding="utf-8")
    raw = {
        "mode": mode, "model": {"name": "fake-model", "base_url": "http://localhost:8000/v1", "api_key": "secret-for-test"},
        "scene_name": "test_scene", "task_id": "sample",
        "dataset": {"image_dir": "images"}, "output_dir": "results",
        "backend": backend, "labels": [{"name": "target", "description": "special appearance"}],
        'require_box_reason': False,  # Preserve legacy-response tests; new feedback tests exercise the default.
        "reference_image_dir": "references", "prompt_file": "prompt.txt", "review_prompt_file": "review.txt",
        "max_reference_examples": 0, "max_reference_crops": 0, "max_reference_negatives": 0,
        "inference": {"concurrency": 1, "max_attempts": 1, "enable_completeness_review": False},
    }
    path = root / "config.json"
    write_json(path, raw)
    return path, raw



def make_image(path: Path, size: tuple[int, int] = (640, 360)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (180, 180, 60)).save(path)
    return path



def make_reference_pair(
    directory: Path,
    name: str,
    label: str = LABEL,
    image_path_value: str | None = None,
) -> tuple[Path, Path]:
    image_path = make_image(directory / f"{name}.jpg")
    document = {
        "version": "6.3.1",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[10.0, 20.0], [300.0, 320.0]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_path_value or f"{name}.jpg",
        "imageData": None,
        "imageHeight": 360,
        "imageWidth": 640,
    }
    json_path = directory / f"{name}.json"
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return image_path, json_path



def reference_config(root: Path, **overrides) -> dict:
    ref = root / "references"
    ref.mkdir(exist_ok=True)
    return {
        "reference_image_dir": str(ref), "max_reference_examples": 6,
        "max_reference_crops": 6, "max_reference_negatives": 6,
        "labels": [{"name": LABEL, "description": "reference object"}], **overrides,
    }
