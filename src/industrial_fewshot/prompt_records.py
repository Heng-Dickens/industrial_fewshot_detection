"""Export actual model prompts and associate saved inputs with their metrics."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
import uuid

LOGGER = logging.getLogger(__name__)
INDEX_NAME = 'prompt_results.md'
UNAVAILABLE_TEXT = '历史运行未保存可核验的文字 prompt；无法从当前配置可靠恢复。\n'
REFERENCE_KINDS = ('reference_examples', 'reference_crops', 'reference_negatives',
                   'reference_originals', 'reference_boxed_images')


def _snapshot(backend, model, text, source, references=(), review_template=''):
    kind = 'text_visual' if backend == 'yoloe_fusion' else 'visual' if backend == 'yoloe' else 'none' if backend == 'yolo_onnx' else 'text'
    if kind in {'text', 'text_visual'} and not isinstance(text, str):
        kind = 'unknown'
    return dict(schema='industrial_fewshot.prompt.v1', backend=backend, model=model, kind=kind,
                text=text, text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text, str) else None,
                text_source=source, references=list(references) if kind in {'visual', 'text_visual'} else None,
                review_template=review_template)


def capture_prompt(target: dict) -> dict:
    """Read the prepared runtime target, never a file that may change mid-run."""
    backend, config = target['backend'], target['config']
    if backend in {'pet_dino', 'wedetect'}:
        visual = config[backend].get('prompt_mode') == 'visual'
        text, source = (None, 'support_images_and_boxes') if visual else (config[backend]['text_prompt'], f'{backend}.text_prompt')
    elif backend in {'groundingdino', 'yoloe_text', 'yoloe_fusion'}:
        text, source = config[backend]['text_prompt'], f'{backend}.text_prompt'
    elif backend in {'yoloe', 'yolo_onnx'}:
        text, source = None, 'not_a_text_prompt'
    else:
        text, source = target['prompt'], 'rendered_prompt'
    references = [{key: row[key] for key in ('sample_id', 'image_path', 'image_sha256', 'bboxes', 'crops') if key in row}
                  for row in config.get('reference_examples', [])] if backend in {'yoloe', 'yoloe_fusion', 'pet_dino'} else []
    snapshot = _snapshot(backend, config['model'], text, source, references, target.get('review_template', ''))
    if backend == 'pet_dino' and visual:
        snapshot.update(kind='visual', references=references)
    if backend == 'yoloe_fusion':
        snapshot['fusion'] = {key: config[backend][key] for key in ('visual_weight', 'text_weight', 'positive_count', 'aggregation')}
        snapshot['fusion']['visual_input'] = 'prepared_target_crops'
    snapshot['reference_inputs'] = dict(mode=config.get('reference_mode'),
        counts={key:len(config.get(key, [])) for key in REFERENCE_KINDS},
        manifest_sha256=config.get('reference_manifest_sha256'))
    return snapshot


def recover_prompt(directory: Path, target: dict) -> dict:
    """Recover history from its own saved evidence, without loading current config."""
    if target.get('prompt_snapshot'):
        snapshot = copy.deepcopy(target['prompt_snapshot'])
        if snapshot['kind'] in {'text', 'text_visual'}:
            actual = hashlib.sha256(snapshot['text'].encode('utf-8')).hexdigest()
            if actual != snapshot['text_sha256']:
                raise ValueError(f'提示词快照校验失败: {directory}')
            path = directory / 'prompt.txt'
            if path.is_file():
                existing = path.read_text(encoding='utf-8-sig')
                if existing.strip() and existing != snapshot['text']:
                    raise ValueError(f'提示词文件与该次运行快照不一致: {path}')
        return snapshot
    backend = target.get('backend', 'generic')
    review_path = directory / 'review_template.txt'
    review = review_path.read_text(encoding='utf-8-sig') if review_path.is_file() else ''
    if backend in {'groundingdino', 'yoloe_text', 'yoloe_fusion'}:
        text = target.get('local_model', {}).get('settings', {}).get('text_prompt')
        source = 'saved_run.local_model.settings.text_prompt'
    elif backend in {'yoloe', 'yolo_onnx'}:
        text, source = None, 'saved_run.backend'
    else:
        path = directory / 'prompt.txt'
        text = path.read_text(encoding='utf-8-sig') if path.is_file() else None
        text = text if text and text.strip() and text != UNAVAILABLE_TEXT else None
        source = 'saved_prompt.txt' if text is not None else 'historical_text_unavailable'
        if text is not None and target.get('prompt_sha256'):
            actual = hashlib.sha256((text + review).encode('utf-8')).hexdigest()
            if actual != target['prompt_sha256']:
                raise ValueError(f'历史提示词与运行指纹不一致，拒绝归因到该次结果: {path}')
    references = [{'sample_id': sample} for sample in target.get('reference_sample_ids', [])] if backend in {'yoloe', 'yoloe_fusion'} else []
    snapshot = _snapshot(backend, target.get('model', 'unknown'), text, source, references, review)
    if backend == 'yoloe_fusion':
        settings = target.get('local_model', {}).get('settings', {})
        snapshot['fusion'] = {key: settings.get(key) for key in ('visual_weight', 'text_weight', 'positive_count', 'aggregation')}
        snapshot['fusion']['visual_input'] = 'prepared_target_crops'
    snapshot['reference_inputs'] = dict(mode=target.get('reference_mode'), counts=target.get('reference_count', {}),
                                        manifest_sha256=target.get('reference_manifest_sha256'))
    snapshot['request_signature'] = target.get('request_signature')
    return snapshot


def prompt_text(snapshot: dict) -> str:
    if snapshot['kind'] in {'text', 'text_visual'}:
        return snapshot['text']
    if snapshot['kind'] == 'visual':
        name = 'PET-DINO' if snapshot['backend'] == 'pet_dino' else 'YOLOE'
        lines = [f'输入类型：{name} 视觉提示，无文字 prompt。',
                 '模型使用参考原图及其目标框生成视觉原型；本文件仅说明输入方式，不作为模型文字输入。',
                 f"参考样本数：{len(snapshot['references'])}"]
        for row in snapshot['references']:
            lines.append(f"- {row.get('sample_id', '')}" + (f" | {row['image_path']}" if row.get('image_path') else ''))
        return '\n'.join(lines) + '\n'
    if snapshot['kind'] == 'none':
        return '输入类型：监督检测，无文字 prompt。\n模型仅接收待测图像，不接收文字或 few-shot 参考提示。\n'
    return UNAVAILABLE_TEXT


def save_prompt(directory: Path, snapshot: dict, *, recover=False) -> None:
    text = prompt_text(snapshot)
    path = directory / 'prompt.txt'
    write_text = True
    if recover and path.is_file():
        existing = path.read_text(encoding='utf-8-sig')
        if existing.strip() and existing != text:
            raise ValueError(f'已有非空提示词与历史参数不一致，未覆盖: {path}')
        write_text = not existing.strip()
    if write_text:
        path.write_text(text, encoding='utf-8')
    (directory / 'prompt.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prompt_summary(targets: list[dict]) -> str:
    if not targets:
        return ''
    lines = ['【提示词记录】']
    for target in targets:
        backend = target.get('backend', 'generic')
        snapshot = target.get('prompt_snapshot', {})
        if backend == 'yoloe_fusion':
            settings = snapshot.get('fusion') or target.get('local_model', {}).get('settings', {})
            text = snapshot.get('text') or target.get('local_model', {}).get('settings', {}).get('text_prompt')
            lines.append(f"YOLOE 联合提示：裁剪视觉权重={settings.get('visual_weight')}，文字权重={settings.get('text_weight')}；{settings.get('positive_count')}shot。")
            lines.append(f'文字 prompt：{text}')
            continue
        if backend == 'pet_dino' and snapshot.get('kind') == 'visual':
            lines.append('PET-DINO：独立参考图及目标框，Visual-G 视觉提示；不使用待测图真值。')
        elif backend in {'groundingdino', 'yoloe_text', 'pet_dino', 'wedetect'}:
            text = snapshot.get('text') or target.get('local_model', {}).get('settings', {}).get('text_prompt')
            model_name = {'groundingdino': 'GroundingDINO', 'yoloe_text': 'YOLOE', 'pet_dino': 'PET-DINO', 'wedetect': 'WeDetect'}[backend]
            lines.append(f"{model_name} 文字 prompt：{text if text is not None else '历史记录缺失'}")
        elif backend == 'yoloe':
            lines.append('YOLOE：视觉提示，无文字 prompt。')
        elif backend == 'yolo_onnx':
            lines.append('监督检测：仅输入图像，无文字 prompt。')
        else:
            lines.append('MLLM 文字 prompt：完整内容见 prompt.txt，复核模板见 review_template.txt。')
    lines.append('输入快照：prompt.json；具体记录位于本次结果目录。')
    return '\n'.join(lines) + '\n\n'


def _runs(root: Path):
    from .run_discovery import saved_runs
    for path in sorted(saved_runs(root)):
        metadata = json.loads(path.read_text(encoding='utf-8-sig'))
        yield path.parent, metadata


def _atomic_text(path: Path, text: str):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_prompt_index(root: Path) -> list[dict]:
    root = Path(root).resolve()
    rows = []
    for directory, report in _runs(root):
        metrics_path = directory / 'metrics.json'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8-sig')) if metrics_path.is_file() else {}
        for target in report['targets']:
            target_dir = directory if len(report['targets']) == 1 else directory / target['id']
            snapshot = recover_prompt(target_dir, target)
            rows.append(dict(run_dir=directory.relative_to(root).as_posix(), target_id=target['id'],
                backend=target.get('backend', 'generic'), model=target.get('model', 'unknown'), prompt=snapshot,
                request_signature=target.get('request_signature'),
                prompt_file=(target_dir / 'prompt.txt').relative_to(root).as_posix(),
                AP50=metrics.get('mAP50'), AP50_95=metrics.get('mAP50_95'),
                precision=metrics.get('precision'), recall=metrics.get('recall'),
                score_threshold=metrics.get('settings', report.get('evaluation', {})).get('score_threshold'),
                status=metrics.get('evaluation_status', 'not_available')))
    lines = ['# 提示词与结果对照', '',
             '每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。',
             'AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。', '',
             '| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |',
             '|---|---|---|---:|---:|---:|---:|---|---|']
    kinds = {'text': '文字', 'text_visual': '裁剪视觉+文字', 'visual': '视觉参考', 'none': '无文字（监督检测）', 'unknown': '历史文本缺失'}
    percent = lambda value: '-' if value is None else f'{value * 100:.2f}%'
    for row in rows:
        snapshot = row['prompt']
        text = ' '.join((snapshot['text'] or '').split())
        limit = 200 if row['backend'] in {'groundingdino', 'yoloe_text', 'yoloe_fusion'} else 60
        preview = (text[:limit] + ('…' if len(text) > limit else '')).replace('|', '\\|')
        preview = preview or ('历史文本缺失，无法恢复' if snapshot['kind'] == 'unknown' else '不使用文字提示')
        relative = row['run_dir']
        result_link = (Path(relative) / ('evaluation.txt' if (root / relative / 'evaluation.txt').is_file() else 'run.json')).as_posix()
        lines.append(f"| [{relative or '.'}]({result_link}) | {kinds[snapshot['kind']]} | {preview} | "
                     f"{percent(row['AP50'])} | {percent(row['AP50_95'])} | "
                     f"{percent(row['precision'])} / {percent(row['recall'])} | {row['score_threshold']} | {row['status']} | "
                     f"[prompt.txt]({row['prompt_file']}) |")
    _atomic_text(root / INDEX_NAME, '\n'.join(lines) + '\n')
    _atomic_text(root / 'prompt_results.json', json.dumps({'schema': 'industrial_fewshot.prompt_results.v1', 'runs': rows}, ensure_ascii=False, indent=2) + '\n')
    return rows


def update_prompt_index(root: Path) -> None:
    try:
        refresh_prompt_index(root)
    except (OSError, ValueError, KeyError) as exc:
        LOGGER.warning('提示词结果索引更新失败，单次运行记录已保存：%s', exc)


def backfill_prompts(root: Path, *, dry_run=False) -> list[dict]:
    """Fill derived prompt artifacts; leave inference metadata/data/cache untouched."""
    root = Path(root).resolve()
    plans, jobs = [], []
    for directory, report in _runs(root):
        snapshots = []
        for target in report['targets']:
            target_dir = directory if len(report['targets']) == 1 else directory / target['id']
            snapshot = recover_prompt(target_dir, target)
            snapshots.append({**target, 'prompt_snapshot': snapshot})
            path = target_dir / 'prompt.txt'
            if path.is_file():
                existing = path.read_text(encoding='utf-8-sig')
                if existing.strip() and existing != prompt_text(snapshot):
                    raise ValueError(f'已有提示词与保存参数不一致，未覆盖: {path}')
            plans.append(dict(run_dir=str(directory), target_id=target['id'], kind=snapshot['kind'], text=snapshot['text']))
            jobs.append(('prompt', target_dir, snapshot))
        jobs.append(('report', directory, snapshots))
    if not dry_run:
        for kind, directory, payload in jobs:
            if kind == 'prompt':
                save_prompt(directory, payload, recover=True)
                continue
            evaluation = directory / 'evaluation.txt'
            if evaluation.is_file():
                contents = evaluation.read_text(encoding='utf-8-sig')
                if '【提示词记录】' not in contents:
                    evaluation.write_text(prompt_summary(payload) + contents, encoding='utf-8')
        refresh_prompt_index(root)
    return plans
