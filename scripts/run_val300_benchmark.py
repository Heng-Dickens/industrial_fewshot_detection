"""Prepare, run, resume or stop the existing-method diaowu_val300 benchmark."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import asyncio
import copy
import csv
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from industrial_fewshot.cache import image_sha256, digest
from industrial_fewshot.config import load_config
from industrial_fewshot.configuration import PATH_KEYS
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runner import run
from industrial_fewshot.runtime import stop_owned_process

MANIFEST = ROOT / 'configs/specific/diaowu_val300_benchmark.yaml'
LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def absolute_paths(value, base):
    if isinstance(value, list):
        return [absolute_paths(item, base) for item in value]
    if isinstance(value, dict):
        return {key: str((base / child).resolve()) if key in PATH_KEYS and isinstance(child, str) and child
                else absolute_paths(child, base) for key, child in value.items()}
    return value


def settings():
    document = yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))
    output = (MANIFEST.parent / document['output_root']).resolve()
    return document, output


def prepare():
    document, output = settings()
    output.mkdir(parents=True, exist_ok=True)
    runtime_dir = ROOT / '.planning/val300_configs'
    runtime_dir.mkdir(exist_ok=True)
    dataset = (MANIFEST.parent / document['dataset_config']).resolve()
    jobs, expected_hash = [], None
    for queue in ('local', 'remote'):
        for spec in document[queue]:
            path = runtime_dir / (spec['id'] + '.yaml')
            overlay = {'extends': [str((MANIFEST.parent / spec['config']).resolve()), str(dataset)],
                       'mode': 'train', 'output_dir': str(output), 'scene_name': 'diaowu_val300',
                       'evaluation': document['evaluation'], 'reuse_predictions': True}
            overlay.update(absolute_paths(spec.get('overrides', {}), MANIFEST.parent))
            path.write_text(yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True), encoding='utf-8')
            config = load_config(path)
            # A later edit to a shared prompt must not change queued jobs.
            if queue == 'remote':
                target = config['targets'][0]
                for key in ('prompt_file', 'review_prompt_file'):
                    frozen = runtime_dir / (spec['id'] + '_' + key + '.txt')
                    frozen.write_bytes(Path(target[key]).read_bytes())
                    overlay[key] = str(frozen)
                path.write_text(yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True), encoding='utf-8')
                config = load_config(path)
            assert all('max' not in target['config']['model'].lower() for target in config['targets'])
            suite = config['experiment_suite']['enabled']
            dry = asyncio.run(run_ablation(config, dry_run=True) if suite else run(config, dry_run=True))
            groups = dry['groups'] if suite else [dry]
            for group in groups:
                assert group['image_count'] == 300
                assert group['excluded_reference_overlap'] == []
                assert group['ground_truth_json_image_count'] == 300
                if expected_hash is None:
                    expected_hash = group['dataset_sha256']
                assert group['dataset_sha256'] == expected_hash
                if group.get('experiment', {}).get('positive_count'):
                    assert group['experiment']['positive_count'] == 5
            planned = dry['comparison_dir'] if suite else dry['run_dir']
            jobs.append({'id': spec['id'], 'queue': queue, 'config': str(path),
                         'config_sha256': image_sha256(path), 'status': 'ready',
                         'execution_signature': digest({'dataset': groups[0]['dataset_sha256'],
                             'evaluation': config['evaluation'], 'requests': [[t['request_signature'] for t in group['targets']] for group in groups]}),
                         'model': config['targets'][0]['config']['model'],
                         'backend': config['targets'][0]['backend'], 'expected_groups': len(groups),
                         'expected_predictions': len(groups)*300, 'planned_output': planned,
                         'log': str(output / 'logs' / (spec['id'] + '.log'))})
            print('PREFLIGHT', spec['id'], len(groups), 'groups', len(groups)*300, 'images', flush=True)
    plan = {'created_at': now(), 'manifest': str(MANIFEST), 'manifest_sha256': image_sha256(MANIFEST),
            'dataset_config': str(dataset), 'dataset_sha256': expected_hash, 'image_count': 300,
            'ground_truth_count': 145, 'excluded_models': ['qwen3.8-max'], 'jobs': jobs,
            'expected_group_count': sum(job['expected_groups'] for job in jobs),
            'expected_prediction_count': sum(job['expected_predictions'] for job in jobs)}
    write_json(output / 'benchmark_plan.json', plan)
    write_json(output / 'benchmark_preflight.json', {'plan': plan, 'jobs': jobs, 'all_cohorts_match': True})
    print('PLAN', str(output / 'benchmark_plan.json'), flush=True)
    return plan


def summarize(output, plan):
    from industrial_fewshot.run_discovery import saved_runs
    rows = []
    paths = sorted(path.parent / 'metrics.json' for path in saved_runs(output) if (path.parent / 'metrics.json').is_file())
    for path in paths:
        if not any(path.parent == Path(job['planned_output']) or path.parent.parent == Path(job['planned_output'])
                   for job in plan['jobs']):
            continue
        if not (path.parent / 'run.json').is_file():
            continue
        metadata, metrics = read_json(path.parent / 'run.json'), read_json(path)
        if metadata['image_count'] != 300 or metadata['dataset_sha256'] != plan['dataset_sha256']:
            raise ValueError(f'Inconsistent evaluation cohort: {path.parent}')
        assert all('max' not in target['model'].lower() for target in metadata['targets'])
        row = {'method': path.parent.relative_to(output).as_posix(),
               'status': metrics['evaluation_status'], 'failed_images': metrics['failed_request_count'],
               'AP50_percent': metrics['coco']['stats']['AP50']*100,
               'AP50_95_percent': metrics['coco']['stats']['AP']*100}
        for threshold in ('0.5', '0.7'):
            point = metrics['score_threshold_results'][threshold]
            for key in ('tp', 'fp', 'fn'):
                row[f'{threshold}_{key}'] = point[key]
            for key in ('precision', 'recall', 'f2'):
                row[f'{threshold}_{key}_percent'] = point[key]*100
        row['run_dir'] = str(path.parent)
        rows.append(row)
    write_json(output / 'benchmark_results.json', {'expected_groups': plan['expected_group_count'],
               'evaluated_groups': len(rows), 'image_count': 300, 'ground_truth_count': 145, 'rows': rows})
    lines = ['# diaowu_val300 benchmark', '',
             f"Evaluated groups: {len(rows)} / {plan['expected_group_count']}. Images: 300; GT boxes: 145. Qwen3.8-Max excluded.", '',
             'Native text-only methods use 0-shot; supervised JSTY uses no reference inputs. Visual/fusion/Qwen reference methods use 5-shot.',
             'Existing model parameters are preserved (including the JSTY candidate cutoff 0.5). All methods report score>=0.5 and >=0.7; COCO AP uses all saved candidates.', '',
             '| Method | AP50 | AP50:95 | 0.5 P/R/F2 (%) | 0.7 P/R/F2 (%) | Status |',
             '|---|---:|---:|---|---|---|']
    for row in rows:
        points = [' / '.join(f"{row[f'{threshold}_{key}_percent']:.2f}" for key in ('precision','recall','f2')) for threshold in ('0.5','0.7')]
        lines.append(f"| [{row['method']}]({row['method']}/evaluation.txt) | {row['AP50_percent']:.2f} | {row['AP50_95_percent']:.2f} | {points[0]} | {points[1]} | {row['status']} |")
    (output / 'benchmark_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if rows:
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (output/'benchmark_results.csv').write_text(buffer.getvalue(), encoding='utf-8-sig')
    return len(rows)


def run_batch(output, plan):
    status_path = output/'benchmark_status.json'
    if status_path.is_file():
        old = read_json(status_path)
        if old.get('status') == 'running':
            # Query process existence without sending any signal on Windows.
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, old.get('pid', 0))
            if handle:
                code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(handle)
                if code.value == 259:
                    raise RuntimeError('Benchmark already running; use --status or --stop')
        done = {job['id'] for job in old.get('jobs',[]) if job['status']=='complete'
                and any(p['id'] == job['id'] and p['config_sha256'] == job['config_sha256']
                        and p.get('execution_signature') == job.get('execution_signature') for p in plan['jobs'])}
    else:
        done = set()
    state = {'status':'running','pid':os.getpid(),'started_at':now(),'updated_at':now(),
             'dataset_sha256':plan['dataset_sha256'],'jobs':copy.deepcopy(plan['jobs'])}
    for job in state['jobs']:
        if job['id'] in done:
            job['status']='complete'
            job['planned_output'] = next(previous['planned_output'] for previous in old['jobs'] if previous['id'] == job['id'])
    plan = {**plan, 'jobs': state['jobs']}
    (output/'logs').mkdir(exist_ok=True)
    stop_file = output/'STOP'
    if stop_file.exists():
        stop_file.unlink()
    stop_event = threading.Event()

    def save():
        with LOCK:
            state['updated_at']=now()
            write_json(status_path,state)

    def worker(queue):
        for job in state['jobs']:
            if job['queue']!=queue or job['status']=='complete':continue
            if stop_event.is_set() or stop_file.exists():break
            if image_sha256(Path(job['config']))!=job['config_sha256']:
                raise RuntimeError('Runtime config changed after preflight: '+job['config'])
            command=[sys.executable,str(ROOT/'main.py'),'--config',job['config']]
            env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1',
                 'YOLO_CONFIG_DIR':str(ROOT/'.planning/ultralytics_settings')}
            with open(job['log'],'a',encoding='utf-8') as log:
                log.write('\nBENCHMARK START '+now()+'\n');log.flush()
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,
                    creationflags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)|getattr(subprocess,'CREATE_NO_WINDOW',0))
                with LOCK:
                    job.update(status='running',pid=process.pid,started_at=now(),command=command)
                    save()
                print('START',job['id'],'PID',process.pid,flush=True)
                try:
                    while process.poll() is None:
                        if stop_event.is_set() or stop_file.exists():
                            stop_event.set();stop_owned_process(process);break
                        time.sleep(0.5)
                finally:
                    if process.poll() is None:stop_owned_process(process)
                with LOCK:
                    job.update(status='stopped' if stop_event.is_set() else 'complete' if process.returncode==0 else 'incomplete' if process.returncode==2 else 'failed',
                               exit_code=process.returncode,finished_at=now())
                    save()
                print('FINISH',job['id'],job['status'],flush=True)

    save()
    if hasattr(signal,'SIGBREAK'):
        signal.signal(signal.SIGBREAK,signal.default_int_handler)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,queue) for queue in ('local','remote')]
            try:
                while not all(future.done() for future in futures):
                    for job in state['jobs']:
                        if job['status']=='running' and Path(job['log']).is_file():
                            with open(job['log'],'rb') as log:
                                log.seek(0,2);size=log.tell();log.seek(max(0,size-32000))
                                tail=log.read().decode('utf-8',errors='replace')
                            progress=re.findall(r'diaowu (\d+)/300 ([^\r\n]+)',tail)
                            with LOCK:
                                if progress:job['progress']=progress[-1]
                    save()
                    summarize(output,plan)
                    time.sleep(10)
            except KeyboardInterrupt:
                stop_event.set()
                raise
            finally:
                if stop_file.exists():stop_event.set()
            for future in futures:future.result()
    finally:
        with LOCK:
            state['status']='stopped' if stop_event.is_set() else 'complete' if all(job['status']=='complete' for job in state['jobs']) else 'incomplete'
            state['finished_at']=now();save()
        summarize(output,plan)
    return 0 if state['status']=='complete' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare',action='store_true')
    actions.add_argument('--run',action='store_true')
    actions.add_argument('--status',action='store_true')
    actions.add_argument('--stop',action='store_true')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    _,output=settings()
    if args.prepare:
        prepare();return 0
    if args.stop:
        (output/'STOP').write_text(now(),encoding='utf-8');print('Stop requested; owned workers will stop.');return 0
    if args.status:
        status=read_json(output/'benchmark_status.json')
        print(json.dumps(status,ensure_ascii=False,indent=2));return 0
    plan=read_json(output/'benchmark_plan.json')
    if image_sha256(MANIFEST)!=plan['manifest_sha256']:
        raise RuntimeError('Benchmark manifest changed; run --prepare first')
    return run_batch(output,plan)


if __name__=='__main__':
    raise SystemExit(main())
