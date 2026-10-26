"""Add score 0.5/0.7 reports and additional labels to completed saved runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from industrial_fewshot.evaluation.threshold_outputs import supplement_saved_run
from industrial_fewshot.paths import PROJECT_ROOT


def completed_runs(root):
    from industrial_fewshot.run_discovery import saved_runs
    for path in saved_runs(root):
        if (path.parent / 'metrics.json').is_file():
            yield path.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run-dir', type=Path, nargs='+')
    selection.add_argument('--all-runs', action='store_true')
    parser.add_argument('--report', type=Path, help='Save the batch summary as JSON')
    args = parser.parse_args(argv)
    paths = list(completed_runs(PROJECT_ROOT / 'outputs/runs')) if args.all_runs else args.run_dir
    rows, failed = [], []
    for path in paths:
        try:
            rows.append(supplement_saved_run(path))
            print(f'OK {path}', flush=True)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            failed.append({'run_dir': str(path), 'error': str(exc)})
            print(f'FAILED {path}: {exc}', flush=True)
    result = {'completed': len(rows), 'failed': failed, 'runs': rows}
    if args.report:
        from industrial_fewshot.data.labelme import write_json
        write_json(args.report, result)
    print(json.dumps({'completed': len(rows), 'failed': len(failed)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
