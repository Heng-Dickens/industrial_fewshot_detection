"""Group saved WeDetect prompt experiments under their algorithm; no inference."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from run_wedetect_chache_prompts import (
    ROOT, LABELS, read_plan, reserve_runs, runtime_dir, refresh_method_results,
)
from relocate_wedetect_comparison import PROTECTED, records_digest, verify
from organize_results import checked, relocate, relocate_text
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.prompt_records import refresh_prompt_index


def group_results(output):
    runtime = runtime_dir(output)
    journal = runtime / 'method_layout_migration.json'
    if journal.exists():
        manifest = read_json(journal)
        if manifest['status'] != 'complete':
            raise RuntimeError(f'Inspect incomplete migration before retrying: {journal}')
        verify(manifest)
        return
    old_plan = read_plan(output)
    sources = {group:Path(row['run_dir']) for group,row in old_plan['runs'].items()}
    for source in sources.values():
        assert read_json(source/'summary.json')['failed_request_count'] == 0, source
        assert all((source/label/'metrics.json').is_file() for label in LABELS), source
    plan = reserve_runs(output, copy.deepcopy(old_plan))
    archive = ROOT/'outputs/archive/previous_prompt_layout'/output.name
    backup = ROOT/'.planning'/f'{output.name}_method_layout_backup'
    if backup.exists():
        raise RuntimeError(f'Refusing to overwrite backup: {backup}')
    moves, operations, invariants = {}, [], []
    for group,source in sources.items():
        target = Path(plan['runs'][group]['run_dir'])
        assert not target.exists(), target
        moves[str(source)] = str(target)
        operations.append((source,target))
        for before,after in [(source,target),*[(source/label,target/label) for label in LABELS]]:
            invariants.append(dict(source=str(before),destination=str(after),
                files={p.name:image_sha256(p) for p in before.iterdir() if p.name in PROTECTED and p.is_file()},
                prediction_payload_sha256=records_digest(before/'records.jsonl')))
    for source in dict.fromkeys(path.parent for path in sources.values()):
        target = archive/source.name
        assert not target.exists(), target
        moves[str(source)] = str(target)
        operations.append((source,target))
    for source,target in operations:
        checked(source); checked(target)
    manifest = dict(schema='wedetect.method_layout_migration.v1',status='moving',moves=moves,
        runs={group:row['run_dir'] for group,row in plan['runs'].items()},invariants=invariants,
        backup=str(backup),changed_files=[],policy='Group WeDetect prompts by algorithm; no inference.')
    backup.mkdir(parents=True)
    write_json(backup/'plan.json',manifest)
    write_json(journal,manifest)
    for source,target in operations:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(target))

    def save(path, value):
        data = (json.dumps(value,ensure_ascii=False,indent=2)+'\n') if isinstance(value,dict) else value
        data = data.encode('utf-8')
        if path.exists():
            original = path.read_bytes()
            if original == data:
                return
            saved = backup/'files'/path.relative_to(ROOT)
            saved.parent.mkdir(parents=True,exist_ok=True)
            if not saved.exists():
                saved.write_bytes(original)
        path.write_bytes(data)
        manifest['changed_files'].append(str(path))

    for directory in [*[Path(row['run_dir']) for row in plan['runs'].values()],runtime,output]:
        for path in directory.rglob('*'):
            if (not path.is_file() or path.name in PROTECTED or path.name.endswith('migration.json')
                    or path.name.startswith('prompt_results.')):
                continue
            if path.suffix not in {'.json','.jsonl','.md','.txt'}:
                continue
            text = path.read_text(encoding='utf-8-sig')
            if not any(source in text or Path(source).relative_to(ROOT).as_posix() in text for source in moves):
                continue
            if path.suffix == '.json':
                original = json.loads(text); revised = relocate(original,moves)
                if original != revised:
                    save(path,revised)
            elif path.suffix == '.jsonl':
                original = [json.loads(line) for line in text.splitlines() if line.strip()]
                revised = relocate(original,moves)
                if original != revised:
                    save(path,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in revised))
            else:
                save(path,relocate_text(text,moves,relative=True))
    for group,row in plan['runs'].items():
        directory = Path(row['run_dir'])
        layout = row['result_layout']
        for target in [directory,*[directory/label for label in LABELS]]:
            for filename in ('run.json','summary.json'):
                path = target/filename
                report = read_json(path)
                report.update(result_layout=layout,run_dir=str(target),output_dir=str(target.parent))
                save(path,report)
        version = Path(layout['prompt_registry_dir'])/'prompt_version.json'
        save(directory/'prompt_version.json',read_json(version))
    save(runtime/'plan.json',plan)
    comparison = read_json(output/'comparison.json')
    comparison['plan'] = {k:v for k,v in plan.items() if k!='items'}
    comparison['groups'] = [dict(id=g,run_dir=r['run_dir']) for g,r in plan['runs'].items()]
    save(output/'comparison.json',comparison)
    save(output/'run_index.json',dict(runs=plan['runs'],runtime_dir=str(runtime),migration=str(journal)))
    path = output/'report.md'
    save(path,path.read_text().replace('outputs/runs/chache_val300/文字提示_vN/wedetect_base_0shot/',
        'outputs/runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_vN/'))
    method_dir = Path(next(iter(plan['runs'].values()))['run_dir']).parent
    for root in [ROOT/'outputs/runs',method_dir.parent,method_dir]:
        refresh_prompt_index(root)
    refresh_method_results(method_dir)
    verify(manifest)
    manifest['status']='complete'
    write_json(journal,manifest)
    print(f'METHOD RESULTS: {method_dir}/experiment_results.md')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/comparisons/wedetect_chache_prompts_20261027')
    parser.add_argument('--apply',action='store_true',required=True)
    args=parser.parse_args()
    group_results(args.output.resolve())
