"""Readable dataset, method and experiment namespaces for project results."""
from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import hashlib
import os
from pathlib import Path

from .cache import digest
from .data.labelme import read_json, write_json
from .outputs import safe_name
from .paths import PROJECT_ROOT
from .prompt_records import capture_prompt

RUNS_ROOT = PROJECT_ROOT / 'outputs/runs'


@contextmanager
def _layout_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.layout.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _namespace(root, stem, filename, identity, payload, *, create, version=False):
    """Atomically reserve readable names; changed contents always get a new version."""
    number = 1
    while True:
        name = f'{stem}_v{number}' if version else stem if number == 1 else f'{stem}_第{number}套'
        directory = root / safe_name(name)
        marker = directory / filename
        if marker.is_file():
            if read_json(marker).get('identity') == identity:
                return directory
        elif not directory.exists():
            if not create:
                return directory
            root.mkdir(parents=True, exist_ok=True)
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            write_json(marker, {'identity': identity, **payload})
            return directory
        number += 1


def namespace_for_report(root: Path, report: dict, *, create=False) -> tuple[Path, dict]:
    """Use saved prompt snapshots and the actual evaluated cohort, never current YAML."""
    with _layout_lock(root) if create else nullcontext():
        return _namespace_for_report(root, report, create=create)


def _namespace_for_report(root: Path, report: dict, *, create=False) -> tuple[Path, dict]:
    count = report['image_count']
    source = Path(report['image_dir'])
    manifest = report.get('dataset_manifest')
    scene = report.get('scene_name', source.parent.name if source.name == 'images' else source.name)
    if manifest:
        scene = Path(manifest).parent.name
    # Keep dataset names explicit while retaining the familiar val300 identifier.
    dataset_name = scene if str(count) in scene else f'{scene}_{count}'
    identity = report['dataset_sha256']
    if root.is_dir():
        for marker in sorted(root.glob('*/dataset.json')):
            if read_json(marker).get('identity') == identity:
                dataset_name = marker.parent.name
                break
    if root.is_dir():
        for marker in sorted(root.glob('*/dataset.json')):
            if read_json(marker).get('identity') == identity:
                dataset_name = marker.parent.name
                break
    dataset_dir = _namespace(root, dataset_name, 'dataset.json', identity,
        {'name': dataset_name, 'image_count': count, 'image_dir': report['image_dir'],
         'ground_truth_dir': report['ground_truth_dir'], 'dataset_manifest': manifest,
         'dataset_sha256': identity}, create=create)
    snapshots = [target['prompt_snapshot'] for target in report['targets']]
    backends = {target['backend'] for target in report['targets']}
    if backends == {'yoloe'} or (backends == {'pet_dino'} and all(s['kind'] == 'visual' for s in snapshots)):
        family = '视觉提示'
    elif backends == {'yolo_onnx'}:
        family = '监督检测'
    elif backends <= {'groundingdino', 'yoloe_text', 'yoloe_fusion', 'pet_dino', 'wedetect'}:
        family = '文字提示'
    elif any(snapshot['kind'] == 'unknown' for snapshot in snapshots):
        family = '历史提示缺失'
    else:
        family = 'MLLM提示'
    contents = [{'text': s.get('text'), 'review_template': s.get('review_template', '')} for s in snapshots]
    prompt_identity = digest({'family': family, 'prompts': contents})
    texts = [s.get('text') or '' for s in snapshots]
    description = ('允许少量遮挡；主体结构及连接关系可辨认时继续标注，逐实例判断。'
                   if any('【遮挡规则：允许少量遮挡，逐个目标判断】' in text for text in texts)
                   else '旧版严格可见规则：人物遮挡时整图不标；保留用于与新版对比。'
                   if any('整张图不标注' in text for text in texts)
                   else '本次实际输入，完整文字及复核模板见各运行目录的 prompt.txt / review_template.txt。')
    prompt_payload = {'family': family, 'description': description, 'prompts': contents,
        'text_sha256': [hashlib.sha256(s['text'].encode()).hexdigest() if isinstance(s.get('text'), str) else None for s in snapshots]}
    extra = {}
    if backends in ({'pet_dino'}, {'yoloe'}) and family == '视觉提示':
        backend = next(iter(backends))
        method = {'pet_dino': 'PET-DINO', 'yoloe': 'YOLOE'}[backend]
        prompt_dir = _namespace(dataset_dir, method, 'method.json', digest({'backend': backend}),
            {'name': method, 'backend': backend}, create=create)
        version = family
        extra = {'method': method, 'method_dir': str(prompt_dir)}
    elif backends in ({'wedetect'}, {'groundingdino'}):
        # Keep every prompt variant under its algorithm. The hidden registry
        # reserves versions independently of model weights and repeat numbers.
        backend = next(iter(backends))
        method = {'wedetect': 'WeDetect', 'groundingdino': 'GroundingDINO'}[backend]
        prompt_dir = _namespace(dataset_dir, method, 'method.json', digest({'backend': backend}),
            {'name': method, 'backend': backend}, create=create)
        registry = _namespace(prompt_dir / '.prompts', 'prompt', 'prompt_version.json',
            prompt_identity, prompt_payload, create=create, version=True)
        version = registry.name
        extra = {'method': method, 'method_dir': str(prompt_dir),
                 'prompt_registry_dir': str(registry), 'model_name_suffix': f'_{version}'}
    else:
        prompt_dir = _namespace(dataset_dir, family, 'prompt_version.json', prompt_identity,
            prompt_payload, create=create, version=family not in {'视觉提示', '监督检测'})
        version = prompt_dir.name
    return prompt_dir, {'root': str(root), 'dataset_dir': str(dataset_dir), 'dataset_name': dataset_dir.name,
                        'dataset_sha256': identity, 'prompt_dir': str(prompt_dir),
                        'prompt_version': version, 'prompt_identity': prompt_identity, **extra}


def apply_result_layout(config: dict, items: list[dict], *, create=False) -> dict:
    if not config.get('results_layout') or config.get('result_layout'):
        return config
    report = {'scene_name': config['scene_name'], 'image_count': len(items),
        'image_dir': str(config['image_dir']), 'ground_truth_dir': str(config['ground_truth_dir']),
        'dataset_manifest': config.get('dataset_manifest'),
        'dataset_sha256': digest({'images': [{'relative_path': i['relative_path'],
            'image_sha256': i['image_sha256'], 'ground_truth': i['ground_truth']} for i in items]}),
        'targets': [{'backend': t['backend'], 'prompt_snapshot': capture_prompt(t)} for t in config['targets']]}
    root = RUNS_ROOT if config['output_dir'].is_relative_to(RUNS_ROOT) else config['output_dir']
    output, layout = namespace_for_report(root, report, create=create)
    result = copy.copy(config)
    result.update(output_dir=output, result_layout=layout,
                  prediction_cache_dir=config.get('prediction_cache_dir', config['output_dir']),
                  prompt_index_root=root,
                  cache_search_dirs=list(dict.fromkeys([*config.get('cache_search_dirs', []), root, Path(layout['dataset_dir'])])))
    return result
