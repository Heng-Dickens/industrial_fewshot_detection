"""Shared MLLM protocol; no model initialization at import time."""
from __future__ import annotations
import io
import json
import math
import mimetypes
import re
from pathlib import Path
from typing import Any

def get_image_info(image_bytes: bytes) -> tuple[int, int, str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Pillow，请先执行: pip install -r requirements.txt"
        ) from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = image.format or "unknown"
            image.verify()
    except Exception as exc:
        raise ValueError(f"无法读取图片: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("图片宽高无效")
    return width, height, image_format


def image_mime_type(image_path: Path, image_format: str) -> str:
    guessed = mimetypes.guess_type(image_path.name)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    mapping = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "BMP": "image/bmp",
        "WEBP": "image/webp",
    }
    return mapping.get(image_format.upper(), "application/octet-stream")


def usage_as_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return None


async def read_completion(
    response: Any, stream: bool
) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
    """统一读取普通响应和 Chat Completions 异步流。"""
    if not stream:
        if not response.choices:
            raise ValueError("模型响应没有 choices")
        choice = response.choices[0]
        content = choice.message.content
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        return (
            content if isinstance(content, str) else "",
            choice.finish_reason,
            usage_as_dict(getattr(response, "usage", None)),
            reasoning_content if isinstance(reasoning_content, str) else None,
        )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage_dict: dict[str, Any] | None = None
    async for chunk in response:
        chunk_usage = usage_as_dict(getattr(chunk, "usage", None))
        if chunk_usage is not None:
            usage_dict = chunk_usage
        for choice in getattr(chunk, "choices", []):
            delta = choice.delta
            delta_content = getattr(delta, "content", None)
            if isinstance(delta_content, str):
                content_parts.append(delta_content)
            delta_reasoning = getattr(delta, "reasoning_content", None)
            if isinstance(delta_reasoning, str):
                reasoning_parts.append(delta_reasoning)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
    return (
        "".join(content_parts),
        finish_reason,
        usage_dict,
        "".join(reasoning_parts) or None,
    )

