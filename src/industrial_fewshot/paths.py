"""Project locations and compatibility identities for the one-time layout move."""
from pathlib import Path
from functools import lru_cache
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ORIGIN_ROOT = PROJECT_ROOT.with_name("annotationtool")
TEMPLATES_DIR = Path(__file__).resolve().parent / "prompts/templates"
COMMON_CONFIG_DIR = PROJECT_ROOT / 'configs/common'
SPECIFIC_CONFIG_DIR = PROJECT_ROOT / 'configs/specific'
LEGACY_CONFIG_DIR = SPECIFIC_CONFIG_DIR  # Compatibility import for existing tools.
CONFIG_FILE_ALIASES = {
    'configs/models/yoloe_visual.yaml': 'configs/common/diaowu_yoloe_config.json',
    'configs/models/yoloe_text.yaml': 'configs/specific/yoloe_text.yaml',
    'configs/models/groundingdino_text.yaml': 'configs/specific/zero_shot.yaml',
    'configs/models/jsty_hanging_object_onnx.yaml': 'configs/specific/jsty_hanging_object.yaml',
    'configs/models/qwen_mllm.yaml': 'configs/specific/qwen_mllm.yaml',
    'configs/legacy/diaowu_yoloe_config.json': 'configs/common/diaowu_yoloe_config.json',
    'configs/legacy/scene_template.json': 'configs/common/scene_template.json',
    'config/diaowu_yoloe_config.json': 'configs/common/diaowu_yoloe_config.json',
    'config/scene_template.json': 'configs/common/scene_template.json',
    'configs/credentials/README.md': 'docs/guides/API_CREDENTIALS.md',
}
MOVED_DIRECTORIES = {
    "config": "configs/specific", "prompts": "src/industrial_fewshot/prompts/templates",
    "runs": "outputs/archive/runs_legacy",
    "outputs/runs/legacy": "outputs/archive/runs_legacy",
    "outputs/runs/runtime": "outputs/runtime",
    "data/DiaoWuexample_prepared": "data/processed/diaowu/references",
    "data/DiaoGouexample": "data/processed/diaogou/references",
    "data/Chacheexampledata": "data/processed/chache/references",
    'configs/experiments': 'configs/specific',
    'configs/legacy': 'configs/specific',
    'configs/datasets': 'configs/common',
    'configs/credentials': 'configs/common',
}


def resolve_location(path: Path) -> Path:
    path = path.expanduser().resolve()
    if PROJECT_ROOT != CACHE_ORIGIN_ROOT and path.is_relative_to(CACHE_ORIGIN_ROOT):
        path = PROJECT_ROOT / path.relative_to(CACHE_ORIGIN_ROOT)
    if path.exists():
        return path
    manifest = PROJECT_ROOT / 'outputs/result_layout_migration.json'
    if manifest.is_file():
        for old, new in _output_moves(str(manifest), manifest.stat().st_mtime_ns):
            original = Path(old)
            if path == original or path.is_relative_to(original):
                return (Path(new) / path.relative_to(original)).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative in CONFIG_FILE_ALIASES:
            return (PROJECT_ROOT / CONFIG_FILE_ALIASES[relative]).resolve()
    for old, new in MOVED_DIRECTORIES.items():
        original = PROJECT_ROOT / old
        if path == original or path.is_relative_to(original):
            return (PROJECT_ROOT / new / path.relative_to(original)).resolve()
    return path


@lru_cache(maxsize=4)
def _output_moves(filename, modified):
    moves = json.loads(Path(filename).read_text(encoding='utf-8-sig'))['moves']
    return sorted(moves.items(), key=lambda item: len(item[0]), reverse=True)


def stable_project_path(path: Path) -> Path:
    path = path.resolve()
    return CACHE_ORIGIN_ROOT / path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path


def stable_reference_identity(value):
    """Keep old cache identities when only the prepared pool's folder moved."""
    if isinstance(value, list):
        return [stable_reference_identity(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key in {"image_path", "json_path"} and isinstance(child, str):
            path = Path(child)
            for old, new in MOVED_DIRECTORIES.items():
                relocated = PROJECT_ROOT / new
                if path.is_absolute() and path.is_relative_to(relocated):
                    child = str(CACHE_ORIGIN_ROOT / old / path.relative_to(relocated))
                    break
            else:
                if path.is_absolute() and path.is_relative_to(PROJECT_ROOT):
                    child = str(CACHE_ORIGIN_ROOT / path.relative_to(PROJECT_ROOT))
        result[key] = stable_reference_identity(child)
    return result
