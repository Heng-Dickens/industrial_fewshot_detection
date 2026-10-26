from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from industrial_fewshot.configuration import load_document
from industrial_fewshot.data.labelme import write_json
import industrial_fewshot.paths as locations


class ConfigLayoutTest(unittest.TestCase):
    def test_only_common_and_specific_directories_with_flat_configs(self):
        root = locations.PROJECT_ROOT / 'configs'
        self.assertEqual({p.name for p in root.iterdir() if p.is_dir()}, {'common', 'specific'})
        self.assertTrue(all(p.parent.name in {'common', 'specific'} for p in root.rglob('*')
                            if p.is_file() and p.name != 'README.md'))

    def test_old_commands_and_original_project_paths_resolve_to_same_files(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root, old_root = parent / 'industrial_fewshot_detection', parent / 'annotationtool'
            aliases = {
                'configs/experiments/mllm_ablation.yaml': 'configs/specific/mllm_ablation.yaml',
                'configs/models/yoloe_visual.yaml': 'configs/common/diaowu_yoloe_config.json',
                'configs/models/yoloe_text.yaml': 'configs/specific/yoloe_text.yaml',
                'configs/models/groundingdino_text.yaml': 'configs/specific/zero_shot.yaml',
                'configs/models/jsty_hanging_object_onnx.yaml': 'configs/specific/jsty_hanging_object.yaml',
                'configs/legacy/diaowu_yoloe_config.json': 'configs/common/diaowu_yoloe_config.json',
                'configs/datasets/diaowu.yaml': 'configs/common/diaowu.yaml',
                'config/scene_template.json': 'configs/common/scene_template.json',
                'config/annotation_config.json': 'configs/specific/annotation_config.json',
                'config/diaowu_yoloe_config.json': 'configs/common/diaowu_yoloe_config.json',
                'configs/credentials/qwen_mllm.local.json': 'configs/common/qwen_mllm.local.json',
            }
            for relative in set(aliases.values()):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}', encoding='utf-8')
            with patch.object(locations, 'PROJECT_ROOT', root), patch.object(locations, 'CACHE_ORIGIN_ROOT', old_root):
                for old, new in aliases.items():
                    with self.subTest(old=old):
                        self.assertEqual(locations.resolve_location(root / old), root / new)
                        self.assertEqual(locations.resolve_location(old_root / old), root / new)
                external = parent / 'another_project/configs/legacy/a.json'
                self.assertEqual(locations.resolve_location(external), external)

    def test_shared_yoloe_defaults_propagate_to_every_shot_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = root / 'configs/common'
            specific = root / 'configs/specific'
            common.mkdir(parents=True)
            specific.mkdir()
            (common / 'diaowu.yaml').write_text('task_id: diaowu\n', encoding='utf-8')
            base = common / 'diaowu_yoloe_config.json'
            write_json(base, {'backend': 'yoloe', 'yoloe': {'weights': '../../weights/first.pt', 'confidence': 0.001}})
            for count in (1, 5, 10, 25):
                source = locations.PROJECT_ROOT / f'configs/specific/visual_{count}shot.yaml'
                shutil.copyfile(source, specific / source.name)
            for version in ('first', 'second'):
                write_json(base, {'backend': 'yoloe', 'yoloe': {'weights': f'../../weights/{version}.pt', 'confidence': 0.001}})
                for count in (1, 5, 10, 25):
                    raw, _ = load_document(specific / f'visual_{count}shot.yaml')
                    self.assertEqual(raw['yoloe']['positive_count'], count)
                    self.assertEqual(Path(raw['yoloe']['weights']), root / f'weights/{version}.pt')
