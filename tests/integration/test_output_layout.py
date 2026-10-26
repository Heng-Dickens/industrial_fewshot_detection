import asyncio
from pathlib import Path
import shutil
import tempfile
import unittest

from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json
from industrial_fewshot.data.labelme import write_json
from industrial_fewshot.prompt_records import refresh_prompt_index
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.outputs import result_directory
from scripts.run_5shot_batch import comparison_files
from tests.integration.test_reference_suite import suite_fixture
from tests.support import FakeClient, response


class OutputLayoutTest(unittest.TestCase):
    def test_dataset_prompt_versions_history_recovery_and_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = suite_fixture(root, positive_counts=[5], positive_variants=['O+J'], negative_variants=[False])
            raw['results_layout'] = True
            write_json(path, raw)
            config = load_config(path)
            planned = asyncio.run(run_ablation(config, dry_run=True))
            self.assertIn('MLLM提示_v1', Path(planned['comparison_dir']).parts)
            self.assertFalse(config['output_dir'].exists())
            first = asyncio.run(run_ablation(config, client_factory=lambda **_: FakeClient([response()])))
            first_dir = Path(first['comparison_dir'])
            before = {p: p.read_bytes() for p in first_dir.glob('*/records.jsonl')}
            self.assertEqual(len(refresh_prompt_index(config['output_dir'])), 2)
            shutil.rmtree(config['output_dir'] / '.inference_cache')
            # Discover history through dataset / prompt / model / reference-group levels.
            second = asyncio.run(run_ablation(config, client_factory=lambda **_: self.fail('history should be reusable')))
            self.assertEqual(second['reused_prediction_count'], 2)
            resume = {**config, 'resume_comparison_dir': first_dir}
            resumed = asyncio.run(run_ablation(resume, client_factory=lambda **_: self.fail('completed groups must be preserved')))
            self.assertEqual(Path(resumed['comparison_dir']), first_dir)
            for file, contents in before.items():
                self.assertEqual(file.read_bytes(), contents)
            prompt_path = Path(config['targets'][0]['prompt_file'])
            partial = next(first_dir.glob('*/summary.json')).parent
            (partial / 'summary.json').unlink()
            (partial / 'metrics.json').unlink()
            asyncio.run(run_ablation(resume, client_factory=lambda **_: self.fail('partial group should recover from cache')))
            archived = next(config['output_dir'].glob('.interrupted/*/*/*/records.jsonl'))
            self.assertEqual(archived.read_bytes(), before[partial / 'records.jsonl'])
            self.assertTrue((partial / 'metrics.json').is_file())
            prompt_path.write_text(prompt_path.read_text(encoding='utf-8') + '\nAllow identifiable partial occlusion.', encoding='utf-8')
            changed_config = load_config(path)
            changed = asyncio.run(run_ablation(changed_config, dry_run=True))
            self.assertIn('MLLM提示_v2', Path(changed['comparison_dir']).parts)
            self.assertEqual(changed['reused_prediction_count'], 0)
            comparison_before = (first_dir / 'comparison.json').read_bytes()
            with self.assertRaisesRegex(ValueError, '恢复目录'):
                asyncio.run(run_ablation({**changed_config, 'resume_comparison_dir': first_dir}))
            self.assertEqual((first_dir / 'comparison.json').read_bytes(), comparison_before)
            raw['scene_name'] = 'another_dataset'
            write_json(path, raw)
            another = asyncio.run(run_ablation(load_config(path), dry_run=True))
            self.assertEqual(Path(another['comparison_dir']).parents[1], first_dir.parents[1])
            truth_path = root / 'images/frame.json'
            truth = read_json(truth_path)
            truth['shapes'][0]['points'][0][0] += 1
            write_json(truth_path, truth)
            another = asyncio.run(run_ablation(load_config(path), dry_run=True))
            self.assertNotEqual(Path(another['comparison_dir']).parents[1], first_dir.parents[1])

    def test_model_shot_groups_have_direct_results_and_repeat_reuses_cache_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = suite_fixture(root, positive_counts=[5], positive_variants=['O+J'], negative_variants=[False])
            config = load_config(path)
            planned = asyncio.run(run_ablation(config, dry_run=True))
            expected = root / 'results/fake-model_5shot'
            self.assertEqual(Path(planned['comparison_dir']), expected)
            self.assertFalse(expected.exists())
            first = asyncio.run(run_ablation(config, client_factory=lambda **_: FakeClient([response()])))
            self.assertEqual(Path(first['comparison_dir']), expected)
            self.assertEqual({Path(group['run_dir']).name for group in first['groups']},
                             {'仅文字提示', '原图+标注JSON_无负例'})
            for group in first['groups']:
                target = Path(group['run_dir'])
                self.assertEqual(target.parent, expected)
                for file in ('run.json', 'summary.json', 'records.jsonl', 'metrics.json', 'evaluation.txt', 'prompt.txt'):
                    self.assertTrue((target / file).is_file(), file)
                self.assertTrue((target / 'predictions/frame.json').is_file())
                self.assertTrue((target / 'feedback/frame.json').is_file())
                self.assertFalse((target / 'sample').exists())
            comparison_before = (expected / 'comparison.json').read_bytes()
            second = asyncio.run(run_ablation(config, client_factory=lambda **_: self.fail('Expected cache reuse')))
            self.assertEqual(Path(second['comparison_dir']).name, 'fake-model_5shot_第2次')
            self.assertEqual(second['reused_prediction_count'], 2)
            self.assertEqual((expected / 'comparison.json').read_bytes(), comparison_before)
            self.assertEqual(comparison_files(config), {
                expected / 'comparison.json', Path(second['comparison_dir']) / 'comparison.json'})
            shutil.rmtree(root / 'results/.inference_cache')
            third = asyncio.run(run_ablation(config, client_factory=lambda **_: self.fail('Expected history recovery')))
            self.assertEqual(third['reused_prediction_count'], 2)
            self.assertEqual(Path(third['comparison_dir']).name, 'fake-model_5shot_第3次')

    def test_multiple_shots_and_negative_groups_are_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = suite_fixture(root, positive_counts=[1, 5], positive_variants=['O+C+B+J'])
            report = asyncio.run(run_ablation(load_config(path), dry_run=True))
            self.assertEqual(Path(report['comparison_dir']).name, 'fake-model_1shot+5shot')
            names = {Path(group['run_dir']).name for group in report['groups']}
            self.assertIn('5shot_原图+裁剪图+带框图+标注JSON_含5张负例', names)
            self.assertIn('1shot_原图+裁剪图+带框图+标注JSON_无负例', names)
            self.assertEqual(len(names), len(report['groups']))
            self.assertFalse((root / 'results').exists())

    def test_numbering_does_not_overwrite_a_conflicting_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'qwen3.8-27b_5shot').write_text('existing', encoding='utf-8')
            path = result_directory(root, 'qwen3.8-27b_5shot', create=True)
            self.assertEqual(path.name, 'qwen3.8-27b_5shot_第2次')
            self.assertTrue(path.is_dir())
            self.assertEqual((root / 'qwen3.8-27b_5shot').read_text(), 'existing')
