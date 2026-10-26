"""Automatic success-only reuse, keyed by effective request and image contents."""
from __future__ import annotations

import copy
import asyncio
import errno
import hashlib
import json
import logging
import os
from pathlib import Path

from industrial_fewshot.postprocess.schema import validate_prediction
from industrial_fewshot.data.labelme import read_json, write_json, validate_pixel_prediction
from industrial_fewshot.paths import resolve_location, stable_project_path
from industrial_fewshot.labels import target_label_names

LOGGER = logging.getLogger(__name__)
GENERATION_KEYS = (
    "model", "temperature", "top_p", "max_tokens", "enable_thinking", "stream",
    "enable_completeness_review", "structured_output_mode",
)


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def image_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def request_signature(target: dict, cache_version: str = "v1") -> str:
    """Accept both current descriptors and metadata from earlier run.json files."""
    if target["backend"] in {"yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"}:
        return digest({"version": 1, "cache_version": cache_version, "target_id": target["id"],
                       "backend": target["backend"], "label": target["label"], "local_model": target["local_model"],
                       "reference_manifest_sha256": target["reference_manifest_sha256"]})
    generation = target["generation_settings"]
    signature = {
        "version": 1, "cache_version": cache_version,
        "target_id": target["id"], "backend": target["backend"],
        "model": target["model"], "base_url": target["base_url"].rstrip("/"),
        "label": target["label"], "prompt_sha256": target["prompt_sha256"],
        "reference_manifest_sha256": target["reference_manifest_sha256"],
        "generation": {key: generation[key] for key in GENERATION_KEYS if key != "model"},
    }
    if target.get('require_box_reason'):
        signature['require_box_reason'] = True
    if len(target_label_names(target)) > 1:
        signature['labels'] = target_label_names(target)
    if target.get('api_protocol', 'openai_chat_completions') != 'openai_chat_completions':
        signature['api_protocol'] = target['api_protocol']
    if target.get('proxy_request_model'):
        signature['proxy_request_model'] = target['proxy_request_model']
    return digest(signature)


class PredictionCache:
    def __init__(self, output_dir: Path, image_dir: Path, target: dict, cache_version: str = "v1",
                 search_dirs: list[Path] | None = None):
        self.output_dir = output_dir
        self.image_dir = image_dir.resolve()
        self.target = target
        self.cache_version = cache_version
        self.signature = request_signature(target, cache_version)
        self.directory = output_dir / ".inference_cache" / self.signature
        self.search_dirs = list(dict.fromkeys(resolve_location(p) for p in [output_dir, *(search_dirs or [])]))
        self._historical = None

    def _path(self, item: dict) -> Path:
        key = digest({
            "image_root": os.path.normcase(str(stable_project_path(self.image_dir))),
            "relative_path": item["relative_path"], "image_sha256": item["image_sha256"],
        })
        return self.directory / (key + ".json")

    def _validated(self, record: dict, item: dict) -> bool:
        try:
            if (record.get("status") != "success" or record.get("target_id") != self.target["id"]
                    or record.get("relative_path") != item["relative_path"]):
                return False
            validator = validate_pixel_prediction if self.target["backend"] in {"yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"} else validate_prediction
            annotation = validator(record, set(target_label_names(self.target)), item["width"], item["height"])
            if self.target.get('require_box_reason') and any(not obj.get('reason') for obj in annotation['objects']):
                return False
            return all(
                "confidence" in obj and obj["bbox_pixels_xyxy"] == original.get("bbox_pixels_xyxy")
                for obj, original in zip(annotation["objects"], record["objects"])
            )
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError):
            return False

    def _load_historical(self):
        self._historical = {}
        from .run_discovery import saved_runs
        metadata_paths = {p for root in self.search_dirs for p in saved_runs(root)}
        for metadata_path in sorted(metadata_paths, reverse=True):
            try:
                metadata = read_json(metadata_path)
                if resolve_location(Path(metadata["image_dir"])) != self.image_dir:
                    continue
                candidate = next((t for t in metadata["targets"] if t["id"] == self.target["id"]), None)
                if candidate is None:
                    continue
                if request_signature(candidate, metadata.get("cache_version", "v1")) != self.signature:
                    continue
                cutoff = metadata_path.stat().st_mtime_ns
                records_path = metadata_path.parent / "records.jsonl"
                if not records_path.is_file():
                    continue
                for line in records_path.read_text(encoding="utf-8-sig").splitlines():
                    try:
                        record = json.loads(line)
                        if record.get("target_id") == self.target["id"] and record.get("status") == "success":
                            self._historical.setdefault(record["relative_path"], []).append((record, cutoff, str(records_path)))
                    except (ValueError, KeyError, TypeError, AttributeError):
                        # An interrupted append may leave an incomplete last line.
                        continue
            except (OSError, ValueError, KeyError, TypeError, StopIteration, AttributeError):
                continue

    def get(self, item: dict) -> dict | None:
        filename = self._path(item).name
        for root in self.search_dirs:
            path = root / ".inference_cache" / self.signature / filename
            if not path.is_file():
                continue
            try:
                entry = read_json(path)
                record = entry["record"]
                if (entry["request_signature"] == self.signature
                        and entry["image_sha256"] == item["image_sha256"]
                        and self._validated(record, item)):
                    result = copy.deepcopy(record)
                    result.update(cache_hit=True, reused_from=entry.get("source_run_dir", str(path)))
                    return result
            except (OSError, ValueError, KeyError, TypeError):
                LOGGER.warning("缓存记录损坏，尝试从历史运行恢复: %s", path)
        if self._historical is None:
            self._load_historical()
        for record, cutoff, source in self._historical.get(item["relative_path"], []):
            recorded_hash = record.get("image_sha256")
            if recorded_hash is not None:
                if recorded_hash != item["image_sha256"]:
                    continue
            elif item["path"].stat().st_mtime_ns > cutoff:
                # Old records cannot prove the original bytes. Reject images
                # modified after that run; future entries always carry SHA256.
                continue
            if self._validated(record, item):
                result = copy.deepcopy(record)
                result.update(cache_hit=True, reused_from=source, legacy_cache_adoption=recorded_hash is None)
                return result
        return None

    def save(self, item: dict, record: dict, run_dir: Path) -> None:
        if not self._validated(record, item):
            return
        write_json(self._path(item), {
            "request_signature": self.signature, "image_sha256": item["image_sha256"],
            "source_run_dir": str(run_dir), "record": record,
        })

    def lock(self, item: dict):
        return _PredictionLock(self._path(item).with_suffix(".lock"))


class _PredictionLock:
    """OS-owned lock prevents concurrent launches from paying for the same image.

    Locks are released by the OS on process exit; no stale PID cleanup needed.
    """
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    async def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        try:
            while True:
                self.stream.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    await asyncio.sleep(0.2)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    def release(self):
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None
