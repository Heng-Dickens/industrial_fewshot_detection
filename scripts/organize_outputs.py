"""Migrate experiment artifacts to method-first storage, without rerunning models.

Preview by default; --apply moves files and rebases documentation. The manifest
also lets readers resolve paths recorded in immutable historical run metadata.
"""
from pathlib import Path
import argparse
import json
import os
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs/_shared/layout_migration.json'


def plan():
    moves = {}
    def add(old, new):
        if (ROOT / old).exists():
            moves[old] = new
    methods = {'YOLOE': 'YOLOE', 'PET-DINO': 'PET-DINO', 'WeDetect': 'WeDetect',
               'GroundingDINO': 'GroundingDINO', 'OVD-MLLM': 'OVD-MLLM',
               '视觉提示': 'YOLOE', '监督检测': 'JSTY-YOLO', '文字提示_v1': 'WeDetect',
               '文字提示_v2': 'WeDetect'}
    runs = ROOT / 'outputs/runs'
    for dataset in sorted(runs.iterdir()) if runs.exists() else []:
        if not dataset.is_dir() or dataset.name.startswith('.') or dataset.name == 'runtime':
            continue
        for child in sorted(dataset.iterdir()):
            old = str(child.relative_to(ROOT))
            if child.is_dir():
                if child.name.startswith('文字提示'):
                    backend_methods = {'groundingdino':'GroundingDINO', 'wedetect':'WeDetect',
                                       'yoloe_text':'YOLOE', 'yoloe_fusion':'YOLOE'}
                    backends = {t.get('backend') for p in child.rglob('run.json')
                                for t in json.loads(p.read_text(encoding='utf-8-sig')).get('targets', [])}
                    if len(backends) > 1:
                        for item in child.iterdir():
                            if item.is_dir():
                                run = json.loads((item/'run.json').read_text(encoding='utf-8-sig'))
                                method = backend_methods[run['targets'][0]['backend']]
                                add(str(item.relative_to(ROOT)), f'outputs/{method}/experiments/{dataset.name}/{child.name}/{item.name}')
                            else:
                                add(str(item.relative_to(ROOT)), f'outputs/_shared/registries/{dataset.name}/{child.name}/{item.name}')
                        continue
                method = 'Qwen' if child.name.startswith('MLLM') else methods.get(child.name)
                if not method:
                    raise ValueError(f'Unclassified experiment directory: {old}')
                suffix = f'/{child.name}' if child.name.startswith(('MLLM', '文字提示')) or child.name == '视觉提示' else ''
                add(old, f'outputs/{method}/experiments/{dataset.name}{suffix}')
            else:
                if 'comparison' in child.name or child.name.startswith('qwen'):
                    add(old, f'outputs/Qwen/comparisons/{dataset.name}/{child.name}')
                else:
                    add(old, f'outputs/_shared/datasets/{dataset.name}/{child.name}')
    for scene in ('chache', 'diaowu'):
        add(f'outputs/finetuning/{scene}_yoloe', f'outputs/YOLOE/finetune/{scene}_val300/runs')
        add(f'examples/yoloe_finetune_{scene}', f'outputs/YOLOE/finetune/scripts/{scene}')
        add(f'outputs/comparisons/finetuning/{scene}_val300/yoloe_search_20261029',
            f'outputs/YOLOE/finetune/{scene}_val300/comparisons/yoloe_search_20261029')
    for old, new in {
        'examples/yoloe_finetune_common': 'outputs/YOLOE/finetune/scripts/common',
        'examples/FINETUNING.md': 'outputs/YOLOE/finetune/README.md',
        'examples/ovd_mllm_filtering': 'outputs/OVD-MLLM/scripts',
        'examples/chat_image.py': 'outputs/Qwen/scripts/chat_image.py',
        'examples/chat_qwen38_image.py': 'outputs/Qwen/scripts/chat_qwen38_image.py',
        'outputs/comparisons/fewshot/cross_dataset/yoloe_qwen_filtering_20261030':
            'outputs/OVD-MLLM/comparisons/yoloe_qwen_filtering_20261030',
        'outputs/comparisons/fewshot/cross_dataset/pet_dino_shot_sweep_20261027':
            'outputs/PET-DINO/comparisons/pet_dino_shot_sweep_20261027',
        'outputs/comparisons/fewshot/chache_val300/wedetect_prompt_sweep_20261027':
            'outputs/WeDetect/comparisons/chache_val300/wedetect_prompt_sweep_20261027',
        'outputs/comparisons/fewshot/cross_dataset/aligned_baselines_20261027':
            'outputs/_shared/comparisons/aligned_baselines_20261027',
        'outputs/comparisons/localization/cross_dataset/iou85_box20_20261028':
            'outputs/_shared/comparisons/iou85_box20_20261028',
        'outputs/comparisons/archive': 'outputs/_shared/archive/comparisons',
        'outputs/archive': 'outputs/_shared/archive/history',
        'outputs/runtime': 'outputs/_shared/runtime',
        'outputs/environment_setup': 'outputs/_shared/environment_setup',
        'outputs/runs/.inference_cache': 'outputs/_shared/cache/inference',
        'outputs/runs/runtime': 'outputs/_shared/legacy_run_runtime',
    }.items():
        add(old, new)
    for directory in ('outputs/runs', 'outputs/comparisons'):
        base = ROOT / directory
        for child in sorted(base.iterdir()) if base.exists() else []:
            if child.is_file():
                add(str(child.relative_to(ROOT)), f'outputs/_shared/archive/layout_indexes/{base.name}/{child.name}')
    return moves


def remap(path, moves):
    for old, new in sorted(moves.items(), key=lambda item: len(item[0]), reverse=True):
        base = ROOT / old
        if path == base or path.is_relative_to(base):
            return ROOT / new / path.relative_to(base)
    return path


def rebase_links(text, old_file, new_file, moves):
    def replace(match):
        raw = match[2]
        target, sep, title = raw.partition(' "')
        target = target.strip('<>')
        if not target or target.startswith('#') or urlsplit(target).scheme:
            return match[0]
        path, fragment_sep, fragment = target.partition('#')
        original = (old_file.parent / unquote(path)).resolve()
        resolved = remap(original, moves)
        value = str(resolved) if path.startswith('/') else os.path.relpath(resolved, new_file.parent)
        if ' ' in value:
            value = '<' + value + '>'
        if fragment_sep:
            value += '#' + fragment
        return match[1] + value + (sep + title if sep else '') + match[3]
    return re.sub(r'(\[[^\]\n]*\]\()([^\n)]+)(\))', replace, text)


def inventory(path):
    files = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [path]
    return {str(p.relative_to(path)) if path.is_dir() else '.':
            (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if MANIFEST.exists():
        print('Method-first migration is already recorded; no files moved.')
        return
    moves = plan()
    print(json.dumps(moves, ensure_ascii=False, indent=2))
    if not args.apply:
        return
    for old, new in moves.items():
        if (ROOT / new).exists():
            raise FileExistsError(f'Refusing to merge or overwrite: {new}')
    # Record documentation origins before moving, including relative links.
    documents = [ROOT / 'README.md', *(ROOT / 'docs').rglob('*.md'),
                 *(ROOT / 'examples').rglob('*.md'), *(ROOT / 'outputs').rglob('*.md')]
    documents = [p for p in documents if p.is_file() and not any(
        part in {'runtime', 'archive', '.inference_cache', '__pycache__'} for part in p.parts)]
    originals = {p: p.read_text(encoding='utf-8-sig') for p in documents}
    counts = []
    for old, new in sorted(moves.items(), key=lambda item: len(item[0]), reverse=True):
        source, destination = ROOT / old, ROOT / new
        before = inventory(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        if inventory(destination) != before:
            raise RuntimeError(f'Artifact identity changed during move: {new}')
        counts.append(dict(old=old, new=new, files=len(before), bytes=sum(v[1] for v in before.values())))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(schema='industrial_fewshot.method_layout.v1',
        moves={str(ROOT / a): str(ROOT / b) for a, b in moves.items()},
        artifacts=counts, verified='Same inode, byte length and modification time for every moved file before documentation edits.'),
        ensure_ascii=False, indent=2) + '\n')
    changed = 0
    replacements = dict(moves)
    replacements.update({'outputs/runtime': 'outputs/_shared/runtime',
                         'outputs/environment_setup': 'outputs/_shared/environment_setup'})
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
    for old_file, text in originals.items():
        new_file = remap(old_file, moves)
        updated = rebase_links(text, old_file, new_file, moves)
        updated = pattern.sub(lambda m: replacements[m[0]], updated)
        if updated != text:
            new_file.write_text(updated, encoding='utf-8')
            changed += 1
    # The top-level legacy trees must disappear; never leave compatibility symlinks.
    for base in [ROOT/'examples', ROOT/'outputs/runs', ROOT/'outputs/comparisons', ROOT/'outputs/finetuning']:
        if base.exists():
            for directory in sorted((p for p in base.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
                if not any(directory.iterdir()):
                    directory.rmdir()
            if not any(base.iterdir()):
                base.rmdir()
            else:
                raise RuntimeError(f'Unclassified files remain in {base}')
    print(f'Moved {len(counts)} trees/files; rebased {changed} documents. All artifact moves verified.')


if __name__ == '__main__':
    main()
