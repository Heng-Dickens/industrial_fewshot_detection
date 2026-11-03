"""Shared configuration loading for the Chache and Diaowu fine-tuning examples."""
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
DEFAULT_CONFIG = None
SCENES = {'chache': ['Chache1', 'Chache2', 'Chache3', 'Chache4', 'Qianyinche'],
          'diaowu': ['hanging object']}


def load_config(path, chain=()):
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
    protocol = config.get('class_protocol')
    expected = SCENES.get(scene)
    if protocol is not None:
        if (scene != 'chache' or protocol.get('type') != 'heldout_visual_class_v1'
                or protocol.get('base_classes') != ['Chache1', 'Chache2', 'Chache3', 'Qianyinche']
                or protocol.get('novel_classes') != ['Chache4']
                or protocol.get('validation_classes') != protocol['base_classes']
                or protocol.get('training_partition') not in {'base', 'all'}):
            raise ValueError('Invalid explicit held-out class protocol.')
        expected = protocol['base_classes'] if protocol['training_partition'] == 'base' else SCENES[scene]
    if scene not in SCENES or config['class_names'] != expected:
        raise ValueError(f'Class names/order do not match scene {scene!r}.')
    if len(config['text_prompts']) != len(config['class_names']):
        raise ValueError('Text prompt count must match class count.')
    return config
