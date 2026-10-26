from pathlib import Path
import tempfile
import unittest

from industrial_fewshot.data.labelme import read_json, write_json
from scripts.run_val300_benchmark import summarize


class BenchmarkSummaryTest(unittest.TestCase):
    def test_new_prompt_results_do_not_enter_the_old_benchmark_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, ap in (('prompt_v1', 1.0), ('prompt_v2', 0.0)):
                run = root / version / 'fake-model_5shot' / 'original_crop'
                run.mkdir(parents=True)
                write_json(run / 'run.json', {'image_count': 300, 'dataset_sha256': 'same-cohort',
                                             'targets': [{'model': 'fake-model'}]})
                point = {'tp': int(ap), 'fp': int(1-ap), 'fn': int(1-ap), 'precision': ap, 'recall': ap, 'f2': ap}
                write_json(run / 'metrics.json', {'evaluation_status': 'complete', 'failed_request_count': 0,
                    'coco': {'stats': {'AP50': ap, 'AP': ap}}, 'score_threshold_results': {'0.5': point, '0.7': point}})
                (run / 'evaluation.txt').write_text('fixture', encoding='utf-8')
            selected = root / 'prompt_v1/fake-model_5shot'
            count = summarize(root, {'dataset_sha256': 'same-cohort', 'expected_group_count': 1,
                                     'jobs': [{'planned_output': str(selected)}]})
            self.assertEqual(count, 1)
            rows = read_json(root / 'benchmark_results.json')['rows']
            self.assertEqual(rows[0]['AP50_percent'], 100.0)
            self.assertTrue(Path(rows[0]['run_dir']).is_relative_to(selected))
