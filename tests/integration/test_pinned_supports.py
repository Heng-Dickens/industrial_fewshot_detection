import asyncio
import tempfile
import unittest
from pathlib import Path

from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.data.preprocessing import prepare_experiment_dataset
from industrial_fewshot.data.supports import ReferencePool
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runner import run
from tests.integration.test_yoloe_backend import local_fixture


def pinned_fixture(root):
    path, raw = local_fixture(root)
    raw.update(preparation={'prepared_dir': 'processed', 'splits_dir': 'splits', 'supports_dir': 'supports',
                            'support_counts': [1, 5, 10, 25], 'support_seeds': [0]}, support_manifest_dir='supports')
    write_json(path, raw)
    prepare_experiment_dataset(load_config(path))
    full = read_json(root / 'supports/10shot/seed_0.json')
    chosen = {key: list(reversed(full[key])) for key in ('positive_samples', 'negative_samples')}
    for count in (1, 5, 10):
        manifest = read_json(root / f'supports/{count}shot/seed_0.json')
        manifest.update(selection_policy='explicit', selection_provenance={'source': 'historical-test'},
                        **{key: rows[:count] for key, rows in chosen.items()})
        write_json(root / f'supports/{count}shot/seed_0.json', manifest)
    return path, raw, chosen


class PinnedSupportTests(unittest.TestCase):
    def test_mllm_and_yoloe_use_same_pinned_originals_and_negative_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw, chosen = pinned_fixture(root)
            local = asyncio.run(run(load_config(path), dry_run=True))
            expected = [row['sample_id'] for row in chosen['positive_samples'][:5]]
            self.assertEqual(local['targets'][0]['reference_sample_ids'], expected)
            self.assertEqual(local['targets'][0]['reference_count']['reference_negatives'], 0)
            raw.update(backend='generic', experiment_suite={'enabled': True, 'include_prompt_only': False,
                'positive_counts': [1, 5, 10], 'positive_variants': ['O+J'], 'negative_variants': [False, True]})
            write_json(path, raw)
            report = asyncio.run(run_ablation(load_config(path), dry_run=True))
            for group in report['groups']:
                metadata = group['experiment']
                references = metadata['reference_sets'][0]
                self.assertEqual(references['positive_sample_ids'],
                                 [row['sample_id'] for row in chosen['positive_samples'][:metadata['positive_count']]])
                self.assertEqual(references['negative_sample_ids'],
                                 [row['sample_id'] for row in chosen['negative_samples'][:metadata['negative_count']]])

    def test_preparation_preserves_pinned_choices_and_unpinned_25shot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, chosen = pinned_fixture(root)
            full = read_json(root / 'supports/25shot/seed_0.json')
            prepare_experiment_dataset(load_config(path))
            self.assertEqual(read_json(root / 'supports/25shot/seed_0.json'), full)
            for count in (1, 5, 10):
                manifest = read_json(root / f'supports/{count}shot/seed_0.json')
                self.assertEqual(manifest['selection_policy'], 'explicit')
                self.assertEqual(manifest['selection_provenance'], {'source': 'historical-test'})
                self.assertEqual(manifest['positive_samples'], chosen['positive_samples'][:count])
            config = load_config(path)['targets'][0]['config']
            pool = ReferencePool(config, 1)
            positives, negatives = pool.select(5, 5)
            self.assertEqual(positives, pool.positives[:5])
            self.assertEqual(negatives, pool.negatives[:5])

    def test_invalid_pinned_ids_fail_and_changed_source_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = pinned_fixture(root)
            manifest_path = root / 'supports/5shot/seed_0.json'
            original = read_json(manifest_path)
            invalid = read_json(manifest_path)
            invalid['positive_samples'][1] = invalid['positive_samples'][0]
            write_json(manifest_path, invalid)
            with self.assertRaisesRegex(ValueError, 'Explicit support selection'):
                asyncio.run(run(load_config(path), dry_run=True))
            write_json(manifest_path, original)
            original['positive_samples'][0]['image_sha256'] = 'tampered'
            write_json(manifest_path, original)
            with self.assertRaisesRegex(ValueError, 'Support image changed'):
                asyncio.run(run(load_config(path), dry_run=True))
