"""Recover saved prompt artifacts and rebuild the prompt/results index; no inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.prompt_records import backfill_prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'outputs/runs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    plans = backfill_prompts(args.runs_root, dry_run=args.dry_run)
    print(json.dumps({'runs':len(plans), 'input_types':dict(Counter(row['kind'] for row in plans)),
                      'dry_run':args.dry_run,'index':str(args.runs_root / 'prompt_results.md')},ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
