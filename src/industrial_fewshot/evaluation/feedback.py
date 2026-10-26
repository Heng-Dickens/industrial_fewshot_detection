"""Save concise, model-authored visible evidence, aligned with final boxes."""
from pathlib import Path

from industrial_fewshot.data.labelme import write_json


def build_feedback(record: dict, target: dict, width: int, height: int) -> dict:
    supported = target['backend'] not in {'yoloe', 'yoloe_text', 'yoloe_fusion', 'groundingdino', 'yolo_onnx', 'pet_dino', 'wedetect'}
    objects = []
    for index, obj in enumerate(record.get('objects', []), 1):
        reason = obj.get('reason') if supported else None
        objects.append({'index': index, 'label': obj['label'], 'confidence': obj.get('confidence'),
                        'bbox_2d': obj.get('bbox_2d'), 'bbox_pixels_xyxy': obj['bbox_pixels_xyxy'],
                        'reason': reason,
                        'reason_status': 'provided' if reason else 'not_provided' if supported else 'unsupported'})
    return {
        'schema': 'industrial_fewshot.feedback.v1', 'relative_path': record['relative_path'],
        'target_id': target['id'], 'backend': target['backend'], 'model': target['config']['model'],
        'image_width': width, 'image_height': height, 'request_status': record['status'],
        'annotation_status': record.get('annotation_status'), 'reason_code': record.get('reason_code'),
        'image_reason': record.get('reason'), 'objects': objects,
        'feedback_source': 'model_reported_visible_evidence' if supported else 'unsupported_by_detector',
        'cache_hit': bool(record.get('cache_hit')), 'reused_from': record.get('reused_from'),
        'review_performed': record.get('review_performed', False), 'review_error': record.get('review_error'),
        'initial_annotation': record.get('initial_annotation'),
        'initial_raw_response': record.get('initial_raw_response'), 'raw_response': record.get('raw_response'),
        'initial_finish_reason': record.get('initial_finish_reason'), 'finish_reason': record.get('finish_reason'),
        'initial_suspension_gate': record.get('initial_suspension_gate'), 'suspension_gate': record.get('suspension_gate'),
        'attempts': record.get('model_feedback_attempts', []),
        'error': record.get('error'),
    }


def format_feedback(feedback: dict) -> str:
    lines = [f"图像：{feedback['relative_path']}", f"检测器：{feedback['backend']}",
             f"请求状态：{feedback['request_status']}；标注状态：{feedback.get('annotation_status') or '无'}",
             f"整图反馈：{feedback.get('image_reason') or '模型未提供'}"]
    if feedback['feedback_source'] == 'unsupported_by_detector':
        lines.append('此检测器仅输出检测框与分数，不提供自然语言逐框理由。')
    for obj in feedback['objects']:
        score = f"{obj['confidence']:.4f}" if obj['confidence'] is not None else '无'
        lines.extend(['', f"框 {obj['index']}：{obj['label']}；confidence={score}",
                      f"像素坐标 xyxy：{obj['bbox_pixels_xyxy']}",
                      f"选择依据：{obj['reason'] or '未提供逐框理由'}"])
    initial = feedback.get('initial_annotation')
    if initial is not None:
        lines.extend(['', f"首轮整图反馈：{initial['reason']}"])
        for index, obj in enumerate(initial['objects'], 1):
            lines.append(f"首轮框 {index}：{obj['bbox_pixels_xyxy']}；{obj.get('reason') or '未提供逐框理由'}")
    lines.extend(['', f"二次复核：{'已完成' if feedback['review_performed'] else '未完成或未启用'}"])
    if feedback.get('review_error'):
        lines.append(f"复核失败，保留首轮结果：{feedback['review_error']}")
    if feedback.get('error'):
        lines.append(f"错误：{feedback['error']}")
    for event in feedback['attempts']:
        if event['status'] != 'success':
            lines.append(f"尝试 {event['attempt']} / {event['stage']}：{event['status']}；{event.get('error', '')}")
    if feedback['cache_hit']:
        lines.append('本次复用了已有模型输出，反馈也从同一记录恢复。')
    return '\n'.join(lines) + '\n'


def save_feedback(run_dir: Path, record: dict, target: dict, width: int, height: int, *, target_dir: Path | None = None) -> dict:
    destination = (target_dir if target_dir is not None else run_dir / target['id']) / 'feedback' / Path(record['relative_path']).with_suffix('.json')
    feedback = build_feedback(record, target, width, height)
    write_json(destination, feedback)
    text = destination.with_suffix('.txt')
    text.write_text(format_feedback(feedback), encoding='utf-8')
    return {'feedback_json': str(destination), 'feedback_text': str(text)}
