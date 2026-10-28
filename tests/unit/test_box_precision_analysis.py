"""Boundary/association tests for offline precision and geometry diagnostics."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from scripts.analyze_box_precision import (analyze, box_errors, build_samples, digest,
                                          load_manifest, match_counts, saved_ap, select_ovd_best, tolerance_ok)
from industrial_fewshot.evaluation.detection_metrics import _curve


def obj(box, score=0.9, label='A'):
    return dict(label=label, bbox_pixels_xyxy=box, confidence=score)


class BoxPrecisionTest(unittest.TestCase):
    def test_ovd_best_excludes_mllm_and_failed_runs_and_ranks_by_ap85(self):
        rows = [(1, dict(method='MLLM', eligible_for_comparison=True, run_dir='mllm')),
                (2, dict(method='YOLOE', eligible_for_comparison=True, run_dir='yolo1')),
                (3, dict(method='YOLOE', eligible_for_comparison=True, run_dir='yolo5')),
                (4, dict(method='YOLOE', eligible_for_comparison=False, run_dir='failed')),
                (5, dict(method='WeDetect', eligible_for_comparison=True, run_dir='we'))]
        values = {'mllm': 1., 'yolo1': .4, 'yolo5': .3, 'failed': .9, 'we': .2}

        def metric(path):
            ap = values[path.parent.name]
            return dict(per_class={'A': dict(ground_truth_count=1,
                ap_by_iou={'0.85': ap, '0.9': ap, '0.95': ap})})

        with patch('scripts.analyze_box_precision.read', side_effect=metric):
            selected = select_ovd_best(rows, Path('dataset'))
        self.assertEqual([r['run_dir'] for _, r in selected], ['yolo1', 'we'])

    def test_errors_use_gt_dimensions_and_twenty_percent_is_inclusive(self):
        gt = [0, 0, 100, 50]
        pred = [10, 5, 130, 65]
        self.assertEqual(box_errors(pred, gt), dict(width=.2, height=.2, center_x=.2, center_y=.2))
        self.assertTrue(tolerance_ok(pred, gt))
        self.assertFalse(tolerance_ok([10, 5, 130.001, 65], gt))

    def test_tolerance_is_independent_of_iou85_and_does_not_double_count(self):
        sample = dict(ground_truth=[obj([0, 0, 100, 50])], predictions=[
            obj([10, 5, 130, 65], .9), obj([10, 5, 130, 65], .8)])
        counts = match_counts([sample], ['A'], .5)
        self.assertEqual(counts['prediction_count'], 2)
        self.assertEqual(counts['joint20_count'], 1)
        self.assertEqual(counts['tp85'], 0)
        self.assertEqual(counts['joint20_prediction_rate'], .5)
        self.assertEqual(counts['joint20_gt_coverage'], 1.)

    def test_wrong_class_never_matches_and_high_score_false_positive_counts(self):
        sample = dict(ground_truth=[obj([0, 0, 100, 50])], predictions=[
            obj([0, 0, 100, 50], .95, 'B'), obj([0, 0, 100, 50], .6)])
        counts = match_counts([sample], ['A', 'B'], .7)
        self.assertEqual(counts['prediction_count'], 1)
        self.assertEqual(counts['joint20_count'], 0)
        self.assertEqual(counts['fn85'], 1)
        counts = match_counts([sample], ['A', 'B'], .5)
        self.assertEqual(counts['joint20_count'], 1)

    def test_empty_predictions_keep_gt_and_undefined_prediction_rate(self):
        counts = match_counts([dict(ground_truth=[obj([0, 0, 100, 50])], predictions=[])], ['A'], .5)
        self.assertEqual(counts['joint20_count'], 0)
        self.assertEqual(counts['joint20_gt_coverage'], 0.)
        self.assertIsNone(counts['joint20_prediction_rate'])

    def test_score_bins_are_nonoverlapping_and_maxdets_is_applied_first(self):
        sample = dict(ground_truth=[obj([0, 0, 100, 50])], predictions=[
            obj([0, 0, 100, 50], 1.), obj([0, 0, 100, 50], .9)])
        self.assertEqual(match_counts([sample], ['A'], .9, 1.)['prediction_count'], 1)
        self.assertEqual(match_counts([sample], ['A'], .9, None)['prediction_count'], 2)
        self.assertEqual(match_counts([sample], ['A'], .9, 1., max_dets=1)['prediction_count'], 0)

    def test_macro_ap_uses_gt_classes_and_keeps_missing_values(self):
        metrics = dict(per_class={
            'A': dict(ground_truth_count=2, ap_by_iou={'0.85': .8, '0.9': .6, '0.95': .4}),
            'B': dict(ground_truth_count=20, ap_by_iou={'0.85': .2, '0.9': .2, '0.95': 0}),
            'C': dict(ground_truth_count=0, ap_by_iou={'0.85': None})})
        result = saved_ap(metrics)
        self.assertEqual(result['AP85'], .5)
        self.assertAlmostEqual(result['AP85_95'], (.5 + .4 + .2) / 3)
        del metrics['per_class']['B']['ap_by_iou']['0.85']
        self.assertIsNone(saved_ap(metrics)['AP85'])

    def test_missing_record_is_not_silently_treated_as_an_empty_image(self):
        manifest = [dict(relative_path='a.jpg', image_sha256='a', width=100, height=50, ground_truth=[])]
        run = dict(targets=[dict(id='one')], image_paths=['a.jpg'])
        with self.assertRaisesRegex(ValueError, '不完整'):
            build_samples(manifest, [], run, ['A'])
        with self.assertRaisesRegex(ValueError, '哈希'):
            build_samples(manifest, [dict(relative_path='a.jpg', target_id='one', status='success', image_sha256='bad')], run, ['A'])

    def test_iou_counts_equal_existing_evaluator_on_random_boxes_and_ties(self):
        rng = random.Random(20261028)
        for _ in range(40):
            samples = []
            for _ in range(3):
                truth = [obj([i * 20, 0, i * 20 + 40, 40], label=rng.choice(['A', 'B']))
                         for i in range(rng.randrange(5))]
                predictions = []
                for _ in range(rng.randrange(8)):
                    x = rng.randrange(0, 100)
                    predictions.append(obj([x, 0, x + 40, 40], rng.choice([.5, .7, .9]), rng.choice(['A', 'B'])))
                samples.append(dict(ground_truth=truth, predictions=predictions))
            for score in (0., .5, .7, 1.):
                counts = match_counts(samples, ['A', 'B'], score, max_dets=4)
                for threshold, key in ((.5, 'tp50'), (.85, 'tp85')):
                    expected = sum(_curve(samples, label, threshold, 4, score)['tp'] for label in ('A', 'B'))
                    self.assertEqual(counts[key], expected)

    def test_reports_only_still_exports_ap_and_marks_geometry_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            outputs = project / 'outputs'
            dataset = outputs / 'runs/demo'
            directory = dataset / 'method/run'
            directory.mkdir(parents=True)
            images = [dict(relative_path='a.jpg', image_sha256='a', width=100, height=50,
                           ground_truth=[dict(label='A', bbox_pixels_xyxy=[0., 0., 100., 50.])])]
            identity = digest(dict(images=[{k: s[k] for k in ('relative_path', 'image_sha256', 'ground_truth')} for s in images]))
            manifest_path = project / 'data/raw/demo/manifest.json'
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(dict(samples=images)), encoding='utf-8')
            row = dict(dataset='demo', method='detector', model='x', run_dir='method/run',
                       status='完成', eligible_for_comparison=True, ground_truth_count=1,
                       prompt_version='v1', shot_per_class=1, input='O', negative_references=0)
            summary = dict(dataset=dict(name='demo', dataset_sha256=identity), rows=[row])
            (dataset / 'results_summary.json').write_text(json.dumps(summary), encoding='utf-8')
            metrics = dict(per_class={'A': dict(ground_truth_count=1, ap_by_iou={'0.85': .8, '0.9': .7, '0.95': .6})})
            (directory / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            (directory / 'run.json').write_text(json.dumps(dict(dataset_sha256=identity, targets=[dict(id='one')])), encoding='utf-8')
            data = analyze(outputs, outputs / 'comparisons/test')
            self.assertEqual(data['rows'][0]['AP85'], .8)
            self.assertIn('缺少预测坐标', data['rows'][0]['geometry_status'])
            self.assertEqual(data['rows'][0]['score_thresholds'], {})
            self.assertTrue((outputs / 'comparisons/test/README.md').is_file())
            metrics['mAP50'] = 1.
            metrics['per_class']['A']['ap_by_iou'] = {'0.85': 1., '0.9': 1., '0.95': 1.}
            (directory / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            (directory / 'records.jsonl').write_text(json.dumps(dict(relative_path='a.jpg', target_id='one',
                status='success', image_sha256='a', objects=[obj([0., 0., 100., 50.], .9)])) + '\n', encoding='utf-8')
            data = analyze(outputs, outputs / 'comparisons/test')
            self.assertEqual(data['rows'][0]['geometry_status'], '可计算')
            self.assertEqual(data['rows'][0]['score_thresholds']['0.9']['joint20_count'], 1)
            self.assertEqual(data['rows'][0]['score_thresholds']['1']['prediction_count'], 0)


if __name__ == '__main__':
    unittest.main()
