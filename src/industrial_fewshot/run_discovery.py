"""Discover saved runs without walking prediction images or runtime/cache trees."""
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

EXCLUDED = {'legacy', 'runtime', '.inference_cache', 'logs', 'predictions', 'feedback', 'visualizations', 'test_tmp_reference_suite'}


def saved_runs(root: Path):
    root = Path(root)
    if not root.is_dir():
        return
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        metadata = directory / 'run.json'
        if metadata.is_file():
            yield metadata
            continue
        if depth >= 6:
            continue
        try:
            children = sorted(directory.iterdir(), reverse=True)
        except (PermissionError, FileNotFoundError) as exc:
            LOGGER.debug('Skipping unavailable historical directory %s: %s', directory, exc)
            continue
        for child in children:
            if child.name.startswith('.') or child.name in EXCLUDED or not child.is_dir() or child.is_symlink():
                continue
            pending.append((child, depth + 1))
