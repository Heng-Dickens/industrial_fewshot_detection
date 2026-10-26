import copy
import unittest

from industrial_fewshot.evaluation.detection_metrics import evaluate_detection
from industrial_fewshot.evaluation.reporting import format_metrics, format_report
from tests.unit.test_detection_metrics import obj, sample


class EvaluationReportingTests(unittest.TestCase):
    def test_both_score_thresholds_have_separate_counts_and_one_coco_summary(self):
        truths = [obj([0, 0, 20, 20]), obj([40, 40, 60, 60])]
        predictions = [obj([0, 0, 20, 20], score=.5), obj([40, 40, 60, 60], score=.7),
                       obj([70, 70, 90, 90], score=.6)]
        result = evaluate_detection([sample(truths, predictions)], ['target'], {'score_threshold': .5})
        low, high = (result['score_threshold_results'][key] for key in ('0.5', '0.7'))
        self.assertEqual((low['tp'], low['fp'], low['fn']), (2, 1, 0))
        self.assertEqual((high['tp'], high['fp'], high['fn']), (1, 0, 1))
        self.assertAlmostEqual(low['precision'], 2 / 3)
        self.assertAlmostEqual(high['f2'], 5 / 9)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 1, 0))
        text = format_metrics(result)
        self.assertEqual(text.count('Average Precision'), 6)
        self.assertEqual(text.count('VOC-style Evaluation'), 2)
        self.assertIn('score>=0.5', text)
        self.assertIn('score>=0.7', text)

    def test_official_area_and_recall_summaries_for_perfect_detections(self):
        samples = [sample([obj([0, 0, size, size])], [obj([0, 0, size, size])]) for size in (20, 50, 100)]
        before = copy.deepcopy(samples)
        result = evaluate_detection(samples, ['target'])
        for name, value in result['coco']['stats'].items():
            self.assertAlmostEqual(value, 1.0, msg=name)
        self.assertEqual(samples, before)
        text = format_metrics(result)
        self.assertEqual(text.count('Average Precision'), 6)
        self.assertEqual(text.count('Average Recall'), 6)
        self.assertIn('VOC-style Evaluation', text)

    def test_ar_max_detections_is_real_not_repeated_recall(self):
        first, second = obj([0, 0, 20, 20]), obj([40, 40, 60, 60], score=.8)
        false = obj([70, 70, 90, 90], score=.99)
        result = evaluate_detection([sample([first, second], [false, first, second])], ['target'])
        stats = result['coco']['stats']
        self.assertEqual(stats['AR1'], 0)
        self.assertEqual(stats['AR10'], 1)
        self.assertEqual(stats['AR100'], 1)
        self.assertAlmostEqual(stats['AP50'], 2 / 3)
        self.assertEqual(stats['AP_medium'], -1)

    def test_f2_uses_precision_not_average_precision(self):
        # Highest-scoring FP then one TP of two GT: AP=.5*51/101, P=R=F2=.5.
        truths = [obj([0, 0, 20, 20]), obj([40, 40, 60, 60])]
        predictions = [obj([70, 70, 90, 90], score=.99), truths[0]]
        result = evaluate_detection([sample(truths, predictions)], ['target'])
        row = result['coco']['per_class']['target']
        self.assertAlmostEqual(row['AP50'], .5 * 51 / 101)
        self.assertEqual((row['precision'], row['recall'], row['f2']), (.5, .5, .5))
        self.assertEqual(result['f2'], .5)
        text = format_metrics(result)
        self.assertIn('25.2', text)
        self.assertIn('0.50', text)
        self.assertIn('error=FP/dets', text)

    def test_empty_predictions_and_absent_area_have_different_results(self):
        result = evaluate_detection([sample([obj()])], ['target'])
        self.assertEqual(result['coco']['stats']['AP'], 0)
        self.assertEqual(result['coco']['stats']['AR100'], 0)
        self.assertEqual(result['coco']['stats']['AP_small'], -1)
        self.assertEqual(result['coco']['per_class']['target']['f2'], 0)

    def test_all_negative_dataset_and_class_without_truth(self):
        result = evaluate_detection([sample(predictions=[obj()])], ['target', 'absent'])
        self.assertTrue(all(value == -1 for value in result['coco']['stats'].values()))
        self.assertIsNone(result['coco']['per_class']['target']['AP50'])
        self.assertIn('-1.000', format_metrics(result))

    def test_coco_uses_all_scores_and_standard_100_cap_independent_of_operating_point(self):
        result = evaluate_detection([sample([obj()], [obj(score=.2)])], ['target'],
                                    {'score_threshold': .5, 'max_detections_per_image': 1})
        self.assertAlmostEqual(result['coco']['stats']['AP50'], 1)
        self.assertEqual(result['coco']['per_class']['target']['recall'], 1)
        self.assertEqual(result['recall'], 0)
        self.assertIn('score>=0.5, maxDets=1', format_metrics(result))

    def test_summary_reports_do_not_dump_records_or_fake_dry_run_metrics(self):
        text = format_report({'dry_run': True, 'image_count': 99, 'group_count': 15})
        self.assertNotIn('Average Precision', text)
        self.assertIn('No inference or evaluation performed', text)
        metrics = evaluate_detection([sample([obj()], [obj()])], ['target'])
        report = {'groups': [{'metrics': metrics, 'run_dir': 'first'}, {'metrics': metrics, 'run_dir': 'second'}]}
        text = format_report(report)
        self.assertEqual(text.count('Average Precision'), 12)
        self.assertIn('Groups: 2', text)
