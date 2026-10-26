"""Supervise all 15 MLLM five-shot groups, with persistent status and safe retries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.credentials import resolve_api_key
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.data.supports import group_specs
from industrial_fewshot.runtime import stop_owned_process
from industrial_fewshot.outputs import model_shot_name


def now():
    return datetime.now(timezone.utc).isoformat()


def comparison_files(config):
    from industrial_fewshot.run_discovery import saved_runs
    name = model_shot_name(config)
    candidates = {path.parent.parent / 'comparison.json' for path in saved_runs(config['output_dir'])}
    return {path for path in candidates if path.is_file()
            if path.parent.name == name or path.parent.name.startswith(name + '_第')}


def service_status(config, probe_chat=False):
    import httpx2
    model = config['targets'][0]['model_config']
    key = resolve_api_key(model)
    anthropic = model.get('api_protocol') == 'anthropic_messages'
    probe_chat = probe_chat or anthropic
    try:
        with httpx2.Client(trust_env=False, timeout=15) as client:
            headers = {'Authorization': 'Bearer ' + key}
            if probe_chat:
                body = {'model': model['name'], 'messages': [
                        {'role': 'user', 'content': 'Reply with OK.'}],
                        'max_tokens': 64, 'stream': False, 'enable_thinking': False}
                endpoint = '/chat/completions'
                if anthropic:
                    from industrial_fewshot.detectors.claude_messages import message_body
                    body = message_body({**body, 'temperature': 0, 'top_p': 1}, body['messages'])
                    headers['anthropic-version'] = '2023-06-01'
                    endpoint = '/messages'
                response = client.post(model['base_url'].rstrip('/') + endpoint, headers=headers, json=body)
            else:
                response = client.get(model['base_url'].rstrip('/') + '/models', headers=headers)
            body = response.json()
        if probe_chat:
            ready = response.status_code == 200 and bool(body.get('content') if anthropic else body.get('choices'))
            if anthropic and body.get('model') != model['name']:
                ready = False
            return ready, {'http_status': response.status_code, 'check': 'text_completion',
                           'requested_model': model['name'],
                           'api_protocol': model.get('api_protocol', 'openai_chat_completions')}
        names = [entry['id'] for entry in body.get('data', [])]
        ready = response.status_code == 200 and model['name'] in names
        return ready, {'http_status': response.status_code, 'available_models': names}
    except Exception as exc:
        return False, {'error': type(exc).__name__ + ': ' + str(exc)[:250]}


def verify_results(comparison, expected_groups=15):
    problems = []
    groups = comparison.get('groups', [])
    if len(groups) != expected_groups:
        problems.append(f'Expected {expected_groups} groups, found {len(groups)}')
    for group in groups:
        run_dir = Path(group['run_dir'])
        records = [json.loads(line) for line in (run_dir / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(records) != comparison['image_count']:
            problems.append(f'{run_dir.name}: incomplete record count')
        if ({record['relative_path'] for record in records} != set(comparison['image_paths'])
                or group.get('dataset_sha256') != comparison.get('dataset_sha256')):
            problems.append(f'{run_dir.name}: query cohort mismatch')
        for record in records:
            if record['status'] != 'success':
                problems.append(f"{run_dir.name}: failed {record['relative_path']}")
                continue
            if any(not obj.get('reason') for obj in record['objects']):
                problems.append(f"{run_dir.name}: missing box reason {record['relative_path']}")
            if not Path(record.get('feedback_json', '')).is_file() or not Path(record.get('feedback_text', '')).is_file():
                problems.append(f"{run_dir.name}: missing feedback {record['relative_path']}")
        if group.get('evaluation_status') != 'complete' or 'coco' not in group.get('metrics', {}):
            problems.append(f'{run_dir.name}: incomplete evaluation')
    return problems


def summary_markdown(comparison):
    lines = ['# 5-shot experiment results', '', f"Images per group: {comparison['image_count']}", '',
             '| Experiment | COCO AP50 | COCO AP50:95 | Precision | Recall | F2 | Failed |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in comparison['groups']:
        metrics = group['metrics']
        stats = metrics['coco']['stats']
        values = [stats['AP50'], stats['AP'], metrics['precision'], metrics['recall'], metrics['f2']]
        lines.append('| ' + group.get('experiment', {}).get('name', group.get('experiment', {}).get('id', 'baseline')) + ' | ' +
                     ' | '.join(f'{100 * value:.2f}' for value in values) + f" | {group['failed_request_count']} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--max-passes', type=int, default=6)
    parser.add_argument('--wait-hours', type=float, default=12)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--service-check', choices=('models', 'chat'), default='models',
                        help='chat sends a tiny text-only readiness request to proxies without model listings')
    args = parser.parse_args()
    if args.max_passes < 1 or not 0 < args.wait_hours <= 12 or args.poll_seconds < 1:
        parser.error('Use positive passes/poll seconds and a total time limit up to 12 hours')
    config_path, job_dir = args.config.resolve(), args.job_dir.resolve()
    if not job_dir.is_relative_to(ROOT / 'outputs'):
        parser.error('Job directory must be inside project outputs')
    job_dir.mkdir(parents=True, exist_ok=True)
    state = {'started_utc': now(), 'pid': os.getpid(), 'python': sys.executable,
             'config': str(config_path), 'phase': 'checking', 'passes': []}
    def status(**values):
        state.update(values, updated_utc=now())
        write_json(job_dir / 'status.json', state)
        print(json.dumps(values, ensure_ascii=False), flush=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    config = load_config(config_path)
    specs = group_specs(config['experiment_suite'])
    if len(specs) != 15 or config['experiment_suite']['positive_counts'] != [5]:
        raise ValueError('This supervisor requires the full 15-group five-shot matrix')
    model = config['targets'][0]['model_config']
    approved_service = model['name'], model['base_url'], model.get('api_protocol', 'openai_chat_completions')
    status(model=model['name'], endpoint=model['base_url'], group_count=15)
    while True:
        model = config['targets'][0]['model_config']
        if (model['name'], model['base_url'], model.get('api_protocol', 'openai_chat_completions')) != approved_service:
            status(phase='configuration_changed', finished_utc=now())
            return 5
        ready, service = service_status(config, probe_chat=args.service_check == 'chat')
        status(phase='ready' if ready else 'waiting_for_service', service=service)
        if service.get('http_status') in {401, 403, 404}:
            status(phase='service_rejected', finished_utc=now())
            return 1
        if ready:
            break
        if time.monotonic() >= deadline:
            status(phase='service_unavailable', finished_utc=now())
            return 3
        time.sleep(args.poll_seconds)
        config = load_config(config_path)
    model = config['targets'][0]['model_config']
    status(phase='ready', model=model['name'], endpoint=model['base_url'])
    environment = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
    host = urlparse(model['base_url']).hostname
    environment['NO_PROXY'] = ','.join(filter(None, [environment.get('NO_PROXY'), host]))
    environment['no_proxy'] = environment['NO_PROXY']
    for attempt in range(1, args.max_passes + 1):
        if time.monotonic() >= deadline:
            status(phase='time_limit_reached', finished_utc=now())
            return 4
        before = comparison_files(config)
        stdout, stderr = job_dir / f'pass_{attempt}.out.log', job_dir / f'pass_{attempt}.err.log'
        status(phase='running', current_pass=attempt, stdout=str(stdout), stderr=str(stderr))
        with stdout.open('wb') as out, stderr.open('wb') as err:
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'main.py'), '--config', str(config_path)],
                cwd=ROOT, env=environment, stdout=out, stderr=err,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            state['child_pid'] = process.pid
            status(child_pid=process.pid)
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(process.args, args.wait_hours * 3600)
                    try:
                        code = process.wait(timeout=min(0.2, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except KeyboardInterrupt:
                stop_owned_process(process)
                status(phase='cancelled', finished_utc=now())
                return 130
            except subprocess.TimeoutExpired:
                stop_owned_process(process)
                status(phase='time_limit_reached', finished_utc=now())
                return 4
        candidates = comparison_files(config) - before
        latest = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
        result = read_json(latest) if latest else {}
        state['passes'].append({'pass': attempt, 'exit_code': code, 'comparison': str(latest) if latest else None,
                                'failed_request_count': result.get('failed_request_count')})
        if result.get('service_status_code') in {401, 403, 404}:
            status(phase='service_rejected', comparison=str(latest), finished_utc=now())
            return 1
        if latest and code == 0 and result.get('status') == 'complete':
            problems = verify_results(result)
            if not problems:
                (job_dir / 'RESULTS.md').write_text(summary_markdown(result), encoding='utf-8')
                status(phase='complete', comparison=str(latest), results=str(job_dir / 'RESULTS.md'), finished_utc=now())
                return 0
            status(phase='verification_failed', verification_errors=problems[:30])
        else:
            status(phase='retrying', comparison=str(latest) if latest else None,
                   failed_request_count=result.get('failed_request_count'), exit_code=code)
        if time.monotonic() >= deadline:
            status(phase='time_limit_reached', finished_utc=now())
            return 4
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    status(phase='incomplete', finished_utc=now())
    return 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已停止监督任务（Ctrl+C）；已有结果保留。', file=sys.stderr)
        raise SystemExit(130)
