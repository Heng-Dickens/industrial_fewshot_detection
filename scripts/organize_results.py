"""Migrate saved results to dataset/prompt/model namespaces, preserving inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.prompt_records import recover_prompt, refresh_prompt_index
from industrial_fewshot.result_layout import namespace_for_report
from industrial_fewshot.run_discovery import saved_runs

RUNS = ROOT / 'outputs/runs'
BACKUP = ROOT / '.planning/result_layout_backup'
MANIFEST = ROOT / 'outputs/result_layout_migration.json'


def checked(path, root=ROOT):
    path = path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Path outside the intended workspace: {path}')
    # Do not follow junctions during a recursive move or traversal.
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.exists() and (parent.is_symlink() or bool(getattr(parent.stat(), 'st_file_attributes', 0) & 0x400)):
            raise ValueError(f'Reparse point: {parent}')
    return path


def relocate_text(text, moves, *, relative=False):
    for old, new in sorted(moves.items(), key=lambda item: len(item[0]), reverse=True):
        old_path, new_path = Path(old), Path(new)
        for source, target in ((str(old_path), str(new_path)), (old_path.as_posix(), new_path.as_posix())):
            text = text.replace(source, target)
        if relative:
            text = text.replace(old_path.relative_to(ROOT).as_posix(), new_path.relative_to(ROOT).as_posix())
    return text


def relocate(value, moves):
    if isinstance(value, dict):
        return {key: relocate(child, moves) for key, child in value.items()}
    if isinstance(value, list):
        return [relocate(child, moves) for child in value]
    if isinstance(value, str):
        # Preserve raw model messages and prose; rewrite path-valued strings only.
        if not value.startswith((str(ROOT), ROOT.as_posix())):
            return value
        for old, new in sorted(moves.items(), key=lambda item: len(item[0]), reverse=True):
            for source, target in ((old, new), (Path(old).as_posix(), Path(new).as_posix())):
                if value == source or value.startswith(source + '/') or value.startswith(source + '\\'):
                    return target + value[len(source):]
    return value


def backup_write(path, contents, changes):
    original = path.read_bytes()
    if original == contents:
        return
    backup = BACKUP / path.relative_to(ROOT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise ValueError(f'Backup already exists; migration must not overwrite it: {backup}')
    backup.write_bytes(original)
    path.write_bytes(contents)
    changes.append({'path': str(path), 'backup': str(backup),
                    'before_sha256': hashlib.sha256(original).hexdigest(), 'after_sha256': image_sha256(path)})


def organize():
    if MANIFEST.exists():
        raise ValueError('Migration already recorded; use its manifest instead of migrating twice.')
    status_path = RUNS / 'diaowu_val300/benchmark_status.json'
    if status_path.is_file() and read_json(status_path).get('status') == 'running':
        raise ValueError('Stop the benchmark and wait for stopped status before moving results.')
    paths = list(saved_runs(RUNS))
    units, old_runs, invariant = {}, {}, {}
    for path in paths:
        report = read_json(path)
        targets = copy.deepcopy(report['targets'])
        for target in targets:
            target_dir = path.parent if len(targets) == 1 else path.parent / target['id']
            target['prompt_snapshot'] = recover_prompt(target_dir, target)
        report['targets'] = targets
        old_runs[str(path.parent)] = report
        unit = path.parent.parent if (path.parent.parent / 'comparison.json').is_file() else path.parent
        if unit in units:
            previous = units[unit]['report']
            if (previous['dataset_sha256'] != report['dataset_sha256']
                    or [t['prompt_snapshot']['text_sha256'] for t in previous['targets']]
                    != [t['prompt_snapshot']['text_sha256'] for t in targets]):
                raise ValueError(f'Mixed dataset/prompt versions already exist in: {unit}')
        else:
            units[unit] = {'report': report}
        invariant[str(path.parent)] = {file.name: image_sha256(file)
            for name in ('metrics.json', 'prompt.txt', 'prompt.json', 'review_template.txt')
            if (file := path.parent / name).is_file()}
    # Adopt the known active dataset container; regular namespace creation never
    # claims pre-existing unmarked folders.
    val_reports = [report for report in old_runs.values() if report['image_dir'].replace('\\', '/').endswith('/diaowu_val300/images')]
    val_root = RUNS / 'diaowu_val300'
    if val_reports and val_root.is_dir() and not (val_root / 'dataset.json').exists():
        report = val_reports[0]
        write_json(val_root / 'dataset.json', {'identity': report['dataset_sha256'], 'dataset_sha256': report['dataset_sha256'],
            'name': 'diaowu_val300', 'image_count': 300, 'image_dir': report['image_dir'],
            'ground_truth_dir': report['ground_truth_dir'], 'dataset_manifest': report.get('dataset_manifest')})
    moves, layouts = {}, {}
    for source, entry in sorted(units.items()):
        namespace, layout = namespace_for_report(RUNS, entry['report'], create=True)
        destination = namespace / source.name
        checked(source, RUNS)
        checked(destination, RUNS)
        if destination.exists() or str(destination) in moves.values():
            raise ValueError(f'Refusing a destination collision: {destination}')
        moves[str(source)] = str(destination)
        layouts[str(source)] = layout
    # An interrupted leaf is kept as evidence outside the active result tree.
    # Its success-only cache will refill the same group without duplicate runs.
    interrupted = {}
    for old, report in old_runs.items():
        directory = Path(old)
        if (not (directory / 'metrics.json').is_file() and directory.parent in units
                and report['image_count'] == 300):
            archive = ROOT / 'outputs/archive/interrupted_runs/diaowu_val300' / directory.parent.name / directory.name
            checked(archive)
            if archive.exists():
                raise ValueError(f'Archive destination exists: {archive}')
            interrupted[old] = str(archive)
    # Keep the result root dedicated to datasets; caches stay hidden in place.
    auxiliary = {}
    for name, destination in {'legacy': ROOT / 'outputs/archive/runs_legacy', 'runtime': ROOT / 'outputs/runtime'}.items():
        source = RUNS / name
        if source.exists():
            checked(source, RUNS)
            checked(destination)
            if destination.exists():
                raise ValueError(f'Auxiliary destination exists: {destination}')
            auxiliary[str(source)] = str(destination)
    all_moves = {**moves, **interrupted, **auxiliary}
    # Save the exact plan before the first directory move.
    BACKUP.mkdir(parents=True, exist_ok=True)
    write_json(BACKUP / 'plan.json', {'moves': all_moves, 'invariants': invariant})
    for old, new in moves.items():
        checked(Path(old), RUNS).rename(checked(Path(new), RUNS))
    for old, new in interrupted.items():
        current = Path(relocate(old, moves))
        destination = checked(Path(new))
        destination.parent.mkdir(parents=True, exist_ok=True)
        checked(current, RUNS).rename(destination)
    for old, new in auxiliary.items():
        destination = checked(Path(new))
        destination.parent.mkdir(parents=True, exist_ok=True)
        checked(Path(old), RUNS).rename(destination)
    changes = []
    files = set()
    for new in moves.values():
        directory = Path(new)
        files.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.txt', '.csv', '.md'})
    for new in interrupted.values():
        files.update(p for p in Path(new).rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.txt', '.csv', '.md'})
    for cache in (RUNS / '.inference_cache', val_root / '.inference_cache'):
        if cache.is_dir():
            files.update(cache.glob('*/*.json'))
    files.update(p for p in val_root.iterdir() if p.is_file() and p.suffix in {'.json', '.csv', '.md', '.txt'})
    for path in sorted(files):
        if path.name in {'prompt.txt', 'prompt.json', 'review_template.txt'}:
            continue
        before = path.read_bytes()
        if not any(old.encode() in before or old.replace('\\', '\\\\').encode() in before
                   or Path(old).as_posix().encode() in before for old in all_moves):
            continue
        if path.suffix == '.json':
            original = json.loads(before.decode('utf-8-sig'))
            revised = relocate(original, all_moves)
            if original == revised:
                continue
            after = (json.dumps(revised, ensure_ascii=False, indent=2) + '\n').encode()
        elif path.suffix == '.jsonl':
            lines = []
            for line in before.decode('utf-8-sig').splitlines(keepends=True):
                if not line.strip():
                    lines.append(line)
                    continue
                try:
                    original = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)  # preserve any interrupted trailing write
                    continue
                revised = relocate(original, all_moves)
                lines.append(json.dumps(revised, ensure_ascii=False) + '\n' if original != revised else line)
            after = ''.join(lines).encode()
        else:
            after = relocate_text(before.decode('utf-8-sig'), all_moves).encode()
        backup_write(path, after, changes)
    # Add explicit layout metadata, without altering historical detector inputs.
    for old, report in old_runs.items():
        new = Path(relocate(old, all_moves))
        old_unit = Path(old).parent if Path(old).parent in units else Path(old)
        layout = layouts[str(old_unit)]
        for name in ('run.json', 'summary.json'):
            path = new / name
            if not path.is_file():
                continue
            payload = read_json(path)
            payload.update(result_layout=layout, prompt_index_root=str(RUNS))
            after = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()
            # A prior path rewrite already has its original backup.
            if any(row['path'] == str(path) for row in changes):
                path.write_bytes(after)
                next(row for row in changes if row['path'] == str(path))['after_sha256'] = image_sha256(path)
            else:
                backup_write(path, after, changes)
    # Repair links in maintained documentation and exported summaries.
    documents = [ROOT / 'README.md', ROOT / 'configs/README.md', *list((ROOT / 'docs').rglob('*.md'))]
    documents += list((ROOT / 'outputs/reports').glob('*'))
    for path in documents:
        if not path.is_file() or path.suffix not in {'.md', '.json', '.csv', '.txt'}:
            continue
        original = path.read_bytes()
        if path.suffix == '.json':
            payload = json.loads(original.decode('utf-8-sig'))
            revised = relocate(payload, all_moves)
            after = (json.dumps(revised, ensure_ascii=False, indent=2) + '\n').encode() if revised != payload else original
        else:
            after = relocate_text(original.decode('utf-8-sig'), all_moves, relative=True).encode()
        backup_write(path, after, changes)
    for old, hashes in invariant.items():
        new = Path(relocate(old, all_moves))
        for name, checksum in hashes.items():
            if image_sha256(new / name) != checksum:
                raise ValueError(f'Historical prompt/metric bytes changed: {new / name}')
    roots = {RUNS}
    for layout in layouts.values():
        roots.update((Path(layout['dataset_dir']), Path(layout['prompt_dir'])))
    for root in sorted(roots):
        refresh_prompt_index(root)
    write_json(MANIFEST, {'schema': 'industrial_fewshot.result_layout_migration.v1',
        'run_count': len(old_runs), 'moves': all_moves, 'changes': changes,
        'invariants_verified': invariant,
        'policy': 'Only output-path fields/derived links changed. Saved prompts, metrics, model responses, boxes and confidence preserved. Exact prior modified bytes are backed up.'})
    print(json.dumps({'runs': len(old_runs), 'moved_containers': len(moves), 'archived_interrupted': len(interrupted),
                      'rewritten_files': len(changes), 'manifest': str(MANIFEST)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    organize()


if __name__ == '__main__':
    main()
