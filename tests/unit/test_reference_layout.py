"""Reference relocation preserves identity and safe dataset preparation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from industrial_fewshot import paths
from industrial_fewshot.cache import digest
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import write_json
from industrial_fewshot.data.preprocessing import prepare_experiment_dataset
from tests.integration.test_yoloe_backend import local_fixture


class ReferenceLayoutTests(unittest.TestCase):
    def test_old_paths_resolve_to_current_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'project'
            origin=root.with_name('annotationtool')
            cases={
                'data/processed/diaowu/references/a.png':'data/references/diaowu/a.png',
                'data/DiaoWuexample_prepared/a.png':'data/references/diaowu/a.png',
                'data/supports/diaowu/5shot/seed_0.json':'data/references/diaowu/5shot/seed_0.json',
                'data/supports/chache/Chache1/a.png':'data/references/chache/Chache1/a.png',
                'data/processed/chache/references/a.png':'data/references/chache/review/a.png',
                'data/processed/diaogou/references/a.png':'data/references/diaogou/a.png',
                'data/processed/diaowu/annotations.json':'data/references/diaowu/preparation/annotations.json',
            }
            with patch.object(paths,'PROJECT_ROOT',root),patch.object(paths,'CACHE_ORIGIN_ROOT',origin):
                for old,new in cases.items():
                    current=root/new
                    current.parent.mkdir(parents=True,exist_ok=True)
                    current.write_bytes(b'material')
                    self.assertEqual(paths.resolve_location(root/old),current)
                    self.assertEqual(paths.resolve_location(origin/old),current)
                value=paths.stable_reference_identity({'image_path':str(root/'data/references/diaowu/a.png')})
                self.assertEqual(value['image_path'],str(origin/'data/DiaoWuexample_prepared/a.png'))

    def test_local_digest_preserves_identity_and_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(paths,'PROJECT_ROOT',Path(directory)):
            for schema in ['yoloe_visual.v1','yoloe_crop_text.v1','pet_dino.v1']:
                old=dict(schema=schema,examples=[dict(image_path=str(Path(directory)/'data/processed/diaowu/references/a.png'),
                         image_sha256='before',bboxes=[[1,2,3,4]])])
                new=copy.deepcopy(old)
                new['examples'][0]['image_path']=str(Path(directory)/'data/references/diaowu/a.png')
                expected=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
                self.assertEqual(digest(new),expected)
                new['examples'][0]['image_sha256']='changed'
                self.assertNotEqual(digest(new),expected)

    def test_preparation_shared_root_keeps_assets_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            file,raw=local_fixture(root)
            reference=raw['reference_image_dir']
            raw.update(support_manifest_dir=reference,preparation=dict(prepared_dir=f'{reference}/preparation',
                supports_dir=reference,splits_dir='splits',support_counts=[1,5],support_seeds=[0]))
            write_json(file,raw)
            config=load_config(file)
            manifest=(root/reference/'manifest.json').read_bytes()
            report=prepare_experiment_dataset(config)
            self.assertEqual(report['image_count'],1)
            self.assertEqual((root/reference/'manifest.json').read_bytes(),manifest)
            self.assertTrue((root/reference/'5shot/seed_0.json').is_file())
            for key,unsafe in [('prepared_dir',root/reference/'positive/images'),
                               ('supports_dir',root/reference/'positive'),
                               ('prepared_dir',config['image_dir'])]:
                broken=copy.deepcopy(config)
                broken['dataset_preparation'][key]=str(unsafe)
                with self.subTest(key=key,path=unsafe),self.assertRaises(ValueError):
                    prepare_experiment_dataset(broken,dry_run=True)
