"""Detector boundary: only image inputs, never evaluation truth or split labels."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

from industrial_fewshot.data.labelme import validate_pixel_prediction


@dataclass(frozen=True)
class ImageInput:
    path: Path
    relative_path: str
    width: int
    height: int
    image_sha256: str

    @classmethod
    def from_item(cls, item: dict):
        return cls(**{key: item[key] for key in cls.__dataclass_fields__})

    def as_item(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


class Detector(Protocol):
    async def predict(self, image: ImageInput) -> dict: ...
    async def close(self) -> None: ...


async def infer_local(runner, target: dict, item: dict) -> dict:
    record = {"relative_path": item["relative_path"], "target_id": target["id"], "status": "failed"}
    try:
        data = await asyncio.to_thread(item["path"].read_bytes)
        checksum = hashlib.sha256(data).hexdigest()
        if checksum != item["image_sha256"]:
            raise ValueError("图片在检查后发生变化，请重新运行")
        result = await asyncio.to_thread(runner.predict, data)
        result = validate_pixel_prediction(result, {target["label"]}, item["width"], item["height"])
        return {**record, **result, "status": "success", "image_sha256": checksum,
                "attempts": 1, "usage": None, "review_performed": False}
    except Exception as exc:
        return {**record, "error": str(exc)}


class LocalDetector:
    def __init__(self, runner, target):
        self.runner, self.target = runner, target

    async def predict(self, image: ImageInput) -> dict:
        return await infer_local(self.runner, self.target, image.as_item())

    async def close(self):
        await self.runner.close()
