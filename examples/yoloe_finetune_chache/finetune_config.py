"""Shared configuration loading for the Chache and Diaowu fine-tuning examples."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('config.json')
SCENES = {'chache': ['Chache1', 'Chache2', 'Chache3', 'Chache4', 'Qianyinche'],
          'diaowu': ['hanging object']}


def load_config(path=DEFAULT_CONFIG, chain=()):
    path = Path(path).resolve()
    if path in chain:
        raise ValueError(f'Configuration inheritance cycle: {path}')
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    parent = document.pop('extends', None)
    base = load_config(path.parent / parent, (*chain, path)) if parent else {}

    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
        return result

    config = merge(base, document)
    scene = config.get('scene', 'chache')
    if scene not in SCENES or config['class_names'] != SCENES[scene]:
        raise ValueError(f'Class names/order do not match scene {scene!r}.')
    if len(config['text_prompts']) != len(config['class_names']):
        raise ValueError('Text prompt count must match class count.')
    return config
