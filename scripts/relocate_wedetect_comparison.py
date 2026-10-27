"""Relocate one completed WeDetect comparison into standard dataset/prompt/model runs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from run_wedetect_chache_prompts import ROOT, LABELS, GROUP_NAMES, reserve_runs, runtime_dir
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.prompt_records import refresh_prompt_index
from organize_results import checked, relocate, relocate_text

PATH_FIELDS = {'labelme_json', 'labelme_json_by_score', 'feedback_json', 'feedback_text'}
PROTECTED = {'metrics.json', 'prompt.txt', 'prompt.json', 'review_template.txt'}


def records_digest(path, moves=None):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if moves:
        records = relocate(records, moves)
    return digest([{k:v for k,v in record.items() if k not in PATH_FIELDS} for record in records])


def verify(manifest):
    for invariant in manifest['invariants']:
        directory = Path(invariant['destination'])
        for name, checksum in invariant['files'].items():
            assert image_sha256(directory / name) == checksum, directory / name
        inverse_moves = {new:old for old,new in manifest['moves'].items()}
        assert records_digest(directory / 'records.jsonl', inverse_moves) == invariant['prediction_payload_sha256'], directory
    for group, directory in manifest['runs'].items():
        path = Path(directory)
        metadata = read_json(path / 'run.json')
        assert Path(metadata['run_dir']) == path
        assert len(metadata['targets']) == len(LABELS)
        records = [json.loads(line) for line in (path/'records.jsonl').read_text().splitlines() if line]
        assert len(records) == 1500 and len({(r['relative_path'],r['target_id']) for r in records}) == 1500
        assert all(r['status'] == 'success' for r in records)
        for record in records:
            for field in ('labelme_json', 'feedback_json', 'feedback_text'):
                assert Path(record[field]).is_file(), record[field]
            assert all(Path(p).is_file() for p in record['labelme_json_by_score'].values())
        for label in LABELS:
            assert read_json(path/label/'summary.json')['failed_request_count'] == 0
            assert Path(read_json(path/label/'run.json')['run_dir']) == path/label
    print('VERIFIED: 4 combined runs, 20 class runs, 6000 successful records; prompts, metrics and predictions preserved.')


def migrate(output, finalize=False):
    runtime = runtime_dir(output)
    method_migration = runtime / 'method_layout_migration.json'
    if method_migration.exists():
        manifest = read_json(method_migration)
        assert manifest['status'] == 'complete', method_migration
        verify(manifest)
        return
    migration_path = runtime / 'result_layout_migration.json'
    if migration_path.exists():
        existing = read_json(migration_path)
        if existing.get('status') != 'complete':
            if not finalize:
                raise RuntimeError(f'Previous migration is incomplete; inspect {migration_path} or use --finalize after review')
            for target in existing['moves'].values():
                assert Path(target).exists(), target
            verify(existing)
            files = Path(existing['backup']) / 'files'
            existing['changed_files'] = [str(ROOT/p.relative_to(files)) for p in files.rglob('*') if p.is_file()]
            existing['status'] = 'complete'
            write_json(migration_path, existing)
            return
        verify(existing)
        return
    plan = read_json(output/'plan.json')
    assert read_json(output/'workflow_status.json')['status'] == 'complete'
    backup = ROOT/'.planning'/f'{output.name}_layout_backup'
    if backup.exists():
        raise RuntimeError(f'Backup exists; inspect before migrating again: {backup}')
    original_plan = copy.deepcopy(plan)
    completed = {group:read_json(output/group/'class_runs.json') for group in plan['groups']}
    sources = {}
    invariants = []
    for group in plan['groups']:
        assert set(completed[group]) == set(LABELS)
        sources[group] = {'combined':output/group/'combined',
                          **{label:Path(completed[group][label]) for label in LABELS}}
        for name,directory in sources[group].items():
            assert (directory/'metrics.json').is_file()
            records = [json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines() if line]
            assert len(records) == (1500 if name == 'combined' else 300)
            assert all(r['status'] == 'success' for r in records)
    plan = reserve_runs(output,plan)
    archive = ROOT/'outputs/archive/interrupted_runs'/output.name
    moves = {}
    operations = []
    for group,source_dirs in sources.items():
        destination = Path(plan['runs'][group]['run_dir'])
        if destination.exists():
            raise RuntimeError(f'Destination already exists: {destination}')
        for name,source in source_dirs.items():
            target = destination if name == 'combined' else destination/name
            moves[str(source)] = str(target)
            operations.append((source,target))
            invariants.append(dict(source=str(source), destination=str(target),
                files={p.name:image_sha256(p) for p in source.iterdir() if p.name in PROTECTED and p.is_file()},
                prediction_payload_sha256=records_digest(source/'records.jsonl')))
        source_index=output/group/'class_runs.json'
        target_index=runtime/'indices'/f'{group}.json'
        moves[str(source_index)]=str(target_index); operations.append((source_index,target_index))
        moves[str(output/group)]=str(archive/group); operations.append((output/group,archive/group))
    for name,target in [('configs',runtime/'configs'),('manifests',runtime/'manifests'),('runs',runtime/'cache_legacy')]:
        source=output/name
        if source.exists():
            moves[str(source)]=str(target); operations.append((source,target))
    for source in sorted(output.iterdir()):
        if source.is_file() and (source.name=='plan.json' or source.name.startswith('workflow.')
                                 or source.name=='workflow_status.json' or source.suffix=='.log'):
            target=runtime/'logs'/source.name if source.suffix=='.log' else runtime/source.name
            moves[str(source)]=str(target); operations.append((source,target))
    for source,target in operations:
        checked(source); checked(target)
        if target.exists():
            raise RuntimeError(f'Refusing to overwrite {target}')
    backup.mkdir(parents=True)
    manifest=dict(status='moving', schema='wedetect.result_layout_migration.v1', moves=moves,
        invariants=invariants, runs={g:r['run_dir'] for g,r in plan['runs'].items()}, backup=str(backup),
        changed_files=[], policy='Only paths and derived metadata; no inference.')
    write_json(backup/'plan.json',manifest)
    write_json(migration_path,manifest)
    for source,target in operations:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(target))

    def save(path,contents):
        if isinstance(contents,dict):
            contents=(json.dumps(contents,ensure_ascii=False,indent=2)+'\n').encode()
        elif isinstance(contents,str):
            contents=contents.encode()
        original=path.read_bytes() if path.exists() else None
        if original==contents:
            return
        if original is not None:
            saved=backup/'files'/path.relative_to(ROOT)
            saved.parent.mkdir(parents=True,exist_ok=True)
            if not saved.exists():
                saved.write_bytes(original)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(contents)
        manifest['changed_files'].append(str(path))

    roots=[Path(r['run_dir']) for r in plan['runs'].values()]+[runtime,archive]
    for directory in roots:
        for path in directory.rglob('*'):
            if not path.is_file() or path.name in PROTECTED or path==migration_path:
                continue
            if path.suffix not in {'.json','.jsonl'} and path.name!='evaluation.txt':
                continue
            text=path.read_text(encoding='utf-8-sig')
            if str(output) not in text:
                continue
            if path.suffix=='.json':
                original=json.loads(text); revised=relocate(original,moves)
                if original!=revised: save(path,revised)
            elif path.suffix=='.jsonl':
                original=[json.loads(line) for line in text.splitlines() if line.strip()]
                revised=relocate(original,moves)
                if original!=revised: save(path,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in revised))
            else:
                save(path,relocate_text(text,moves,relative=True))
    for group in plan['groups']:
        directory=Path(plan['runs'][group]['run_dir'])
        layout=plan['runs'][group]['result_layout']
        all_configs={label:str(runtime/'configs'/f'{group}_{label}.local.json') for label in LABELS}
        for label in LABELS:
            config_path=Path(all_configs[label]); config=read_json(config_path)
            config.update(output_dir=str(ROOT/'outputs/runs'),prediction_cache_dir=str(runtime/'cache'),
                          cache_search_dirs=[str(runtime/'cache_legacy'),str(directory)])
            save(config_path,config)
        for subdirectory,label in [(directory,None),*[(directory/l,l) for l in LABELS]]:
            for filename in ('run.json','summary.json'):
                path=subdirectory/filename; report=read_json(path)
                report.update(run_dir=str(subdirectory),output_dir=str(subdirectory.parent),
                              prompt_index_root=str(ROOT/'outputs/runs'),result_layout=layout)
                if label is None:
                    report.update(scene_name='chache_val300',dataset_sha256=plan['dataset_sha256'],
                        dataset_manifest=plan['dataset_manifest'],
                        dataset_manifest_sha256=image_sha256(Path(plan['dataset_manifest'])),
                        config_files=all_configs,original_dataset_sha256=original_plan['dataset_sha256'])
                else:
                    report.update(parent_run_dir=str(directory),evaluation_scope=f'One-vs-rest class: {label}')
                save(path,report)
            if label:
                path=subdirectory/'config.resolved.json'; config=read_json(path)
                config.update(output_dir=str(ROOT/'outputs/runs'),prediction_cache_dir=str(runtime/'cache'),
                              cache_search_dirs=[str(runtime/'cache_legacy'),str(directory)])
                save(path,config)
        write_json(directory/'experiment.json',dict(group=group,description=GROUP_NAMES[group],protocol=plan['protocol'],
            comparison_dir=str(output),class_configs=all_configs,class_run_dirs={l:str(directory/l) for l in LABELS}))
        write_json(directory/'prompt.json',dict(schema='wedetect.multi_query_prompt.v1',prompts=plan['groups'][group],
            snapshots=[t['prompt_snapshot'] for t in read_json(directory/'run.json')['targets']]))
    save(runtime/'plan.json',plan)
    comparison=read_json(output/'comparison.json')
    comparison['plan']={k:v for k,v in plan.items() if k!='items'}
    comparison['groups']=[dict(id=g,run_dir=r['run_dir']) for g,r in plan['runs'].items()]
    save(output/'comparison.json',comparison)
    report_path=output/'report.md'; text=report_path.read_text()
    text=text.replace('完整逐图预测、指标及运行配置在各组的类目录和 `combined/`。',
        '逐图预测、指标和提示词位于 `outputs/runs/chache_val300/文字提示_vN/wedetect_base_0shot/`；'
        '模型目录下的 Chache1～Qianyinche 为分类明细。准备文件、缓存和日志位于 `outputs/runtime/`。')
    text+='\n## 运行目录\n\n'
    for group,r in plan['runs'].items():
        directory=Path(r['run_dir'])
        text+=f'- {group}（{GROUP_NAMES[group]}）：[{directory.relative_to(ROOT)}]({directory})\n'
    save(report_path,text)
    write_json(output/'run_index.json',dict(runs=plan['runs'],runtime_dir=str(runtime),migration=str(migration_path)))
    for root in [ROOT/'outputs/runs',ROOT/'outputs/runs/chache_val300',
                 *[Path(r['run_dir']).parent for r in plan['runs'].values()]]:
        refresh_prompt_index(root)
    verify(manifest)
    manifest['status']='complete'; write_json(migration_path,manifest)
    print(json.dumps({'runs':manifest['runs'],'migration':str(migration_path)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/comparisons/wedetect_chache_prompts_20261027')
    parser.add_argument('--apply',action='store_true',required=True)
    parser.add_argument('--finalize',action='store_true',help='Verify an already moved layout before marking it complete')
    args=parser.parse_args()
    migrate(args.output.resolve(), finalize=args.finalize)
