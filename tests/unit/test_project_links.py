"""Regression coverage for paths that broke when the project was relocated."""
from contextlib import contextmanager
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import industrial_fewshot.paths as locations
from industrial_fewshot.data import exports
from industrial_fewshot.cli import main
from industrial_fewshot.legacy import batch_annotate, review_annotations
from scripts.check_links import broken_links
from tests.tools.test_batch_annotate import make_config as batch_config
from tests.tools.test_review_existing_annotations import make_config as review_config


@contextmanager
def relocated_project():
    with tempfile.TemporaryDirectory() as directory:
        old = Path(directory) / 'annotationtool'
        new = Path(directory) / 'industrial_fewshot_detection'
        (new / 'configs/specific').mkdir(parents=True)
        with patch.object(locations, 'PROJECT_ROOT', new), patch.object(locations, 'CACHE_ORIGIN_ROOT', old):
            yield old, new


class ProjectLinkTests(unittest.TestCase):
    def test_evaluate_help_does_not_require_a_run_directory(self):
        for flag in ('--help', '-h'):
            with self.subTest(flag=flag), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(['evaluate', flag])
                self.assertEqual(caught.exception.code, 0)

    def test_documentation_targets_exist(self):
        self.assertEqual(broken_links(locations.PROJECT_ROOT), [])

    def test_batch_accepts_old_config_and_relocated_image_directory(self):
        with relocated_project() as (old, new):
            image_dir = new / 'data/processed/diaogou/references'
            image_dir.mkdir(parents=True)
            raw = batch_config(image_dir)
            raw['image_dir'] = str(old / 'data/DiaoGouexample')
            (new / 'configs/specific/annotation_config.json').write_text(json.dumps(raw), encoding='utf-8')
            args = SimpleNamespace(config=old / 'config/annotation_config.json', image_dir=None,
                                   output=None, checkpoint=None, concurrency=None)
            self.assertEqual(Path(batch_annotate.load_config(args)['image_dir']), image_dir)

    def test_review_accepts_old_config_and_reference_mapping(self):
        with relocated_project() as (old, new):
            raw = review_config(new)
            references = new / 'data/processed/chache/references'
            references.mkdir(parents=True)
            raw['reference_image_dir'] = str(old / 'data/Chacheexampledata')
            (new / 'configs/specific/review.json').write_text(json.dumps(raw), encoding='utf-8')
            config = review_annotations.load_config(old / 'config/review.json')
            self.assertEqual(Path(config['reference_image_dir']), references)
            self.assertEqual(review_annotations.resolve_reference_path(new, None, old / 'data/Chacheexampledata/a.jpg'),
                             references / 'a.jpg')

    def test_export_accepts_old_config_and_saved_run_input(self):
        with relocated_project() as (old, new):
            expected = new / 'outputs/archive/runs_legacy/scene/annotations.json'
            raw = {'output_json': str(old / 'runs/scene/annotations.json')}
            (new / 'configs/specific/export.json').write_text(json.dumps(raw), encoding='utf-8')
            self.assertEqual(exports.resolve_input_path(SimpleNamespace(input=None, config=old / 'config/export.json')), expected)
            self.assertEqual(exports.resolve_input_path(SimpleNamespace(input=old / 'runs/scene/annotations.json')), expected)

    def test_paths_outside_project_are_not_remapped(self):
        external = Path(tempfile.gettempdir()) / 'unrelated/dataset'
        self.assertEqual(batch_annotate.resolve_path(Path.cwd(), external), external.resolve())


if __name__ == '__main__':
    unittest.main()
