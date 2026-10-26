"""Run and safely resume exactly two Qwen27B groups on the new hanging-object cohort."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runtime import run_async

CONFIG = ROOT / 'configs/specific/diaowu_qwen27b_prompt_5shot.yaml'
RUNTIME = ROOT / 'outputs/runtime/diaowu_qwen27b_prompt_5shot'
STATE = RUNTIME / 'status.json'
DATASET = '470b98531e4a34810e7ee42bf250e39c97480e42427c3d84152e29e4d368f3fc'


def now():
    return datetime.now(timezone.utc).isoformat()


async def execute(dry_run=False):
    config = load_config(CONFIG)
    previous = read_json(STATE) if STATE.is_file() else {}
    if previous:
        saved_directory = Path(previous['comparison_dir'])
        if (saved_directory / 'comparison.json').is_file():
            config['resume_comparison_dir'] = saved_directory
    assert all(t['config']['model'] == 'qwen3.8-27b' for t in config['targets'])
    plan = await run_ablation(config, dry_run=True)
    assert plan['image_count'] == 300 and plan['group_count'] == 2
    assert plan['dataset_sha256'] == DATASET
    assert plan['modes'] == ['prompt_only', 'O_B_k05_pos']
    assert not plan['excluded_reference_overlap']
    assert len(plan['groups'][1]['targets']) == 1
    refs = plan['groups'][1]['targets'][0]['reference_count']
    assert refs['reference_originals'] == 5 and refs['reference_boxed_images'] == 5
    assert not refs['reference_negatives']
    identity = dict(dataset_sha256=DATASET, signatures=[
        g['targets'][0]['request_signature'] for g in plan['groups']])
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME / 'preflight.json', dict(total_image_tasks=600, **identity, plan=plan))
    if dry_run:
        print(json.dumps(dict(total_image_tasks=600, **identity, plan=plan),
                         ensure_ascii=False, indent=2, default=str))
        return
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError('Saved dataset/prompt/reference/model identity differs; preserve old status.')
    directory = Path(previous.get('comparison_dir', plan['comparison_dir']))
    if (directory / 'comparison.json').is_file():
        config['resume_comparison_dir'] = directory
    state = dict(**identity, status='running', started=previous.get('started', now()),
                 updated=now(), comparison_dir=str(directory), config=str(CONFIG), total_image_tasks=600)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_json(STATE, state)
    try:
        result = await run_ablation(config)
        state.update(status=result['status'], comparison_dir=result['comparison_dir'],
                     executed_inference_count=result['executed_inference_count'],
                     reused_prediction_count=result['reused_prediction_count'],
                     failed_request_count=result['failed_request_count'], updated=now())
        if result['status'] != 'complete':
            raise RuntimeError('Some image tasks failed; rerun this command to resume successful predictions.')
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                     else 'failed', error=str(exc), updated=now())
        write_json(STATE, state)
        raise
    write_json(STATE, state)
    print('COMPLETE: ' + str(Path(result['comparison_dir']) / 'comparison.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        run_async(execute(args.dry_run))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        logging.error('Experiment stopped: %s', exc)
        raise SystemExit(1)
