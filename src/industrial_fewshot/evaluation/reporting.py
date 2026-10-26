"""Compact terminal/file reports; structured JSON remains the machine interface."""
from pathlib import Path
import unicodedata

from .coco_metrics import f2_score


def _width(value):
    return sum(2 if unicodedata.east_asian_width(char) in {'W', 'F'} else 1 for char in value)


def _table(headers, rows):
    rows = [[str(cell).replace('\n', ' ').replace('\r', ' ') for cell in row] for row in [headers, *rows]]
    widths = [max(_width(row[index]) for row in rows) for index in range(len(headers))]
    separator = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
    lines = [separator]
    for row in rows:
        lines.append('| ' + ' | '.join(cell + ' ' * (width - _width(cell)) for cell, width in zip(row, widths)) + ' |')
        lines.append(separator)
    return '\n'.join(lines)


def _percent(value):
    return f'{100 * value:.1f}' if value is not None else '-1.0'


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _format_operating_point(metrics):
    settings = metrics['settings']
    lines = ['', '************************* VOC-style Evaluation *************************',
                  f"IoU={settings['iou_threshold']:.2f}, score>={settings['score_threshold']:g}, "
                  f"maxDets={settings['max_detections_per_image']}"]
    categories = metrics['per_class']
    rows = []
    for label, row in categories.items():
        dets = row['tp'] + row['fp']
        rows.append([label, row['tp'], dets, row['ground_truth_count'], _percent(row['precision']),
                     _percent(row['recall']), _percent(f2_score(row['precision'], row['recall'])),
                     f"{row['fp'] / dets:.2f}" if dets else '0.00'])
    rows.append(['mean', sum(row['tp'] for row in categories.values()),
                 sum(row['tp'] + row['fp'] for row in categories.values()),
                 sum(row['ground_truth_count'] for row in categories.values()),
                 _percent(_mean([row['precision'] for row in categories.values()])),
                 _percent(_mean([row['recall'] for row in categories.values()])),
                 _percent(_mean([f2_score(row['precision'], row['recall']) for row in categories.values()])),
                 f"{_mean([row['fp'] / (row['tp'] + row['fp']) if row['tp'] + row['fp'] else 0.0 for row in categories.values()]):.2f}"])
    lines.extend([_table(['class', 'tps', 'dets', 'gts', 'precision', 'recall', 'F2', 'error'], rows),
                  f"TP={metrics['tp']} | FP={metrics['fp']} | FN={metrics['fn']}",
                  'AP/P/R/F2: percent; error=FP/dets; F2=5PR/(4P+R). mean: summed counts, macro-average rates.'])
    return lines


def format_metrics(metrics):
    lines = []
    coco = metrics.get('coco')
    if coco:
        lines.append(coco['summary'])
        categories = coco['per_class']
        rows = [[label, _percent(row['AP50']), _percent(row['recall']), _percent(row['f2'])]
                for label, row in categories.items()]
        rows.append(['mean', *[_percent(_mean([row[key] for row in categories.values()]))
                               for key in ('AP50', 'recall', 'f2')]])
        lines.extend(['', _table(['category', 'AP', 'Recall', 'F2'], rows),
                      'COCO categories: AP50 / Recall50; F2 from TP/FP at IoU=0.50, maxDets=100, all scores.'])
    else:
        lines.append('COCO AP/AR unavailable in this saved report; re-evaluate saved predictions to compute it.')
    lines.extend(_format_operating_point(metrics))
    for point in metrics.get('score_threshold_results', {}).values():
        if point['settings']['score_threshold'] != metrics['settings']['score_threshold']:
            lines.extend(_format_operating_point(point))
    if metrics.get('score_threshold_results'):
        lines.append('Score thresholds filter output boxes and P/R/F2 counts; COCO AP/AR above uses all saved scores.')
    if 'evaluation_status' in metrics:
        lines.append(f"Status: {metrics['evaluation_status']} | images: {metrics['evaluated_image_count']} | "
                     f"failed requests: {metrics.get('failed_request_count', 0)} | "
                     f"image accuracy: {_percent(metrics.get('image_accuracy'))}% | "
                     f"proxy loss: {metrics['loss']['total']:.4f}")
    return '\n'.join(lines)


def format_report(report):
    if report.get('dry_run'):
        return (f"Dry run | images: {report.get('image_count', 0)} | groups: {report.get('group_count', 1)} | "
                f"reused: {report.get('reused_prediction_count', 0)} | "
                f"planned: {report.get('scheduled_inference_count', 0)}\nNo inference or evaluation performed.")
    if 'metrics' in report:
        title = report.get('experiment', {}).get('name') or report.get('experiment', {}).get('id') or report.get('scene_name') or 'Detection evaluation'
        from industrial_fewshot.prompt_records import prompt_summary
        prompt = prompt_summary(report.get('targets', []))
        return f"\n{title}\n{prompt}{format_metrics(report['metrics'])}\n\nResults: {report.get('run_dir', '')}"
    if report.get('groups'):
        blocks = [format_report(group) for group in report['groups']]
        blocks.append(f"\nGroups: {len(report['groups'])} | failed requests: {report.get('failed_request_count', sum(g['metrics'].get('failed_request_count', 0) for g in report['groups']))}")
        blocks.append(f"Results: {report.get('comparison_dir', report.get('output_dir', ''))}")
        return '\n\n'.join(blocks)
    return (f"Images: {report.get('image_count', 0)} | success: {report.get('success_request_count', 0)} | "
            f"failed requests: {report.get('failed_request_count', 0)}\n"
            f"Evaluation: {report.get('evaluation_status', report.get('status', 'not_requested'))}\n"
            f"Results: {report.get('run_dir', report.get('output_dir', ''))}")


def save_text_report(directory, report, filename='evaluation.txt'):
    (Path(directory) / filename).write_text(format_report(report) + '\n', encoding='utf-8')
