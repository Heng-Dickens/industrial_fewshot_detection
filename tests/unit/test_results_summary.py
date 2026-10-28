"""Regression tests for merging saved detector results without changing their scope."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.summarize_results import collect_dataset, csv_bytes, refresh


class ResultsSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.outputs = self.project / 'outputs'
        self.root = self.outputs / 'runs/demo'
        self.root.mkdir(parents=True)
        self.save(self.root / 'dataset.json', dict(name='demo', dataset_sha256='data-hash', image_count=2,
                  dataset_manifest='D:\\old\\repo\\data\\manifest.json'))
        self.save(self.project / 'data/manifest.json', dict(dataset_sha256='data-hash', samples=[
            dict(relative_path='a.jpg', image_sha256='a', ground_truth=[{'label': 'A'}, {'label': 'B'}]),
            dict(relative_path='b.jpg', image_sha256='b', ground_truth=[])]))

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def make_run(self, relative='PET-DINO/model_1shot', targets=('A', 'B'), with_records=True,
                 dataset_hash='data-hash', failed=False):
        path = self.root / relative
        self.save(path / 'run.json', dict(dataset_sha256=dataset_hash, image_count=2,
            targets=[dict(id=t, backend='pet_dino', model='model.pth') for t in targets]))
        self.save(path / 'metrics.json', dict(evaluation_status='incomplete' if failed else 'complete',
            evaluated_image_count=2, failed_image_count=1 if failed else 0,
            mAP50=0., mAP50_95=.123456, per_class={t: dict(ground_truth_count=1, AP50=0.) for t in targets},
            settings=dict(score_threshold=.5, iou_threshold=.5, max_detections_per_image=100),
            precision=0., recall=0., f2=0., tp=0, fp=0, fn=len(targets)))
        if with_records:
            rows = [dict(target_id=t, relative_path=name + '.jpg', image_sha256=name,
                         status='failed' if failed and name == 'a' else 'success')
                    for t in targets for name in ('a', 'b')]
            (path / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        return path

    def test_merged_classes_count_images_once_and_link_child_runs(self):
        path = self.make_run()
        self.make_run('PET-DINO/model_1shot/A', targets=('A',))
        self.make_run('PET-DINO/model_1shot/B', targets=('B',))
        summary = collect_dataset(self.root, self.project)
        self.assertEqual(summary['coverage'], dict(primary_runs=1, component_runs=2, metadata_directories=3))
        row = summary['rows'][0]
        self.assertTrue(row['eligible_for_comparison'])
        self.assertEqual(row['success_images'], 2)
        self.assertEqual(row['record_evidence']['success_requests'], 4)
        self.assertEqual(row['verification'], '逐图内容核验')
        self.assertEqual(len(row['component_runs']), 2)
        self.assertTrue(path.is_dir())

    def test_wrong_data_partial_classes_and_failure_are_retained_but_excluded(self):
        self.make_run('PET-DINO/other_1shot', dataset_hash='other')
        self.make_run('PET-DINO/partial_1shot', targets=('A',))
        self.make_run('PET-DINO/failed_1shot', failed=True)
        rows = collect_dataset(self.root, self.project)['rows']
        self.assertEqual(len(rows), 3)
        self.assertFalse(any(r['eligible_for_comparison'] for r in rows))

    def test_reports_only_do_not_claim_per_image_verification_and_zero_is_preserved(self):
        self.make_run(with_records=False)
        summary = collect_dataset(self.root, self.project)
        row = summary['rows'][0]
        self.assertEqual(row['status'], '完成（元数据）')
        self.assertIsNone(row['success_images'])
        data = csv_bytes([row], ['A', 'B'])
        self.assertNotIn(b'\r\r\n', data)
        exported = next(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        self.assertEqual(float(exported['AP50_pct']), 0.)
        self.assertEqual(exported['precision_0_7_pct'], '')
        self.assertAlmostEqual(float(exported['AP50_95_pct']), 12.3456)

    def test_duplicate_and_corrupt_records_cannot_be_complete(self):
        path = self.make_run()
        file = path / 'records.jsonl'
        with file.open('a', encoding='utf-8') as stream:
            stream.write(file.read_text(encoding='utf-8').splitlines()[0] + '\n{broken\n')
        row = collect_dataset(self.root, self.project)['rows'][0]
        self.assertFalse(row['complete'])
        self.assertEqual(row['record_evidence']['duplicate_records'], 1)
        self.assertEqual(row['record_evidence']['invalid_records'], 1)

    def test_changed_image_hash_and_missing_target_records_prevent_completion(self):
        path = self.make_run()
        file = path / 'records.jsonl'
        records = [json.loads(line) for line in file.read_text().splitlines()]
        records[0]['image_sha256'] = 'changed'
        file.write_text(''.join(json.dumps(r) + '\n' for r in records[:-1]), encoding='utf-8')
        row = collect_dataset(self.root, self.project)['rows'][0]
        self.assertFalse(row['complete'])
        self.assertEqual(row['record_evidence']['hash_mismatches'], 1)
        self.assertEqual(row['success_images'], 1)

    def test_legacy_commands_use_the_unified_summary(self):
        from scripts import summarize_chache_results, summarize_diaowu_results
        for module in (summarize_chache_results, summarize_diaowu_results):
            with patch.object(module, 'refresh') as refresh_command:
                module.main(self.root)
                refresh_command.assert_called_once_with(self.outputs, ['demo'])

    def test_old_edited_summary_is_backed_up_and_rerun_is_byte_identical(self):
        self.make_run()
        original = b'user-edited\r\r\n'
        (self.root / 'results_summary.csv').write_bytes(original)
        refresh(self.outputs)
        backups = list((self.outputs / 'archive').rglob('results_summary.csv'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        before = {p.name: p.read_bytes() for p in self.root.glob('results_summary.*')}
        refresh(self.outputs)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.glob('results_summary.*')})


if __name__ == '__main__':
    unittest.main()
