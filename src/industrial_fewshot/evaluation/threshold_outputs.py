"""Supplement saved runs with score-threshold results, without model inference."""
from pathlib import Path

from PIL import Image

from .compare import load_saved_group
from .detection_metrics import evaluation_settings, output_score_thresholds
from .evaluator import evaluate_saved_run
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.data.prediction_outputs import export_prediction_thresholds
from industrial_fewshot.paths import resolve_location
from industrial_fewshot.data.labelme import read_json, write_json


def refresh_comparison_thresholds(directory: Path) -> None:
    path = directory / 'comparison.json'
    if not path.is_file():
        return
    report = read_json(path)
    updated = False
    for group in report.get('groups', []):
        metrics_path = resolve_location(Path(group['run_dir'])) / 'metrics.json'
        if not metrics_path.is_file():
            candidate = directory / Path(group['run_dir']).name
            if (candidate / 'run.json').is_file():
                metadata = read_json(candidate / 'run.json')
                expected = [(t['id'], t.get('request_signature')) for t in group['targets']]
                actual = [(t['id'], t.get('request_signature')) for t in metadata['targets']]
                if (group.get('dataset_sha256') and group['dataset_sha256'] == metadata.get('dataset_sha256')
                        and expected == actual and all(signature for _, signature in expected)):
                    old_path = group['run_dir']
                    group['run_dir'] = str(candidate.resolve())
                    for row in report.get('rows', []):
                        if row.get('run_dir') == old_path:
                            row['run_dir'] = group['run_dir']
                    metrics_path = candidate / 'metrics.json'
        if metrics_path.is_file():
            group['metrics'] = read_json(metrics_path)
            updated = True
    if updated:
        report['comparison_dir'] = str(directory.resolve())
        write_json(path, report)
        from .reporting import save_text_report
        save_text_report(directory, report, 'comparison.txt')


def supplement_saved_run(run_dir: Path) -> dict:
    group = load_saved_group(run_dir)
    source = resolve_location(Path(group['image_dir']))
    settings = evaluation_settings(group.get('evaluation'))
    items = {}
    for record in group['saved_records']:
        relative = record['relative_path']
        if relative not in items:
            path = (source / relative).resolve()
            if not path.is_relative_to(source) or image_sha256(path) != record.get('image_sha256'):
                raise ValueError(f'Query image differs from saved record: {path}')
            with Image.open(path) as image:
                width, height = image.size
            items[relative] = {'path': path, 'relative_path': relative, 'width': width, 'height': height}
    # Validates all saved boxes/GT, updates derived metrics and report only.
    report = evaluate_saved_run(Path(group['run_dir']))
    for record in group['saved_records']:
        if record['status'] != 'success':
            continue
        target_dir = Path(group['run_dir'])
        if len(group['targets']) > 1:
            target_dir /= record['target_id']
        export_prediction_thresholds(target_dir, items[record['relative_path']], record, settings,
                                     include_primary=False)
    refresh_comparison_thresholds(Path(group['run_dir']).parent)
    return {'run_dir': group['run_dir'], 'primary_score_threshold': settings['score_threshold'],
            'score_thresholds': output_score_thresholds(settings),
            'score_threshold_results': report['metrics']['score_threshold_results'],
            'AP50': report['metrics']['coco']['stats']['AP50'],
            'policy': 'Reused saved candidates; raw records and primary predictions unchanged; no inference.'}
