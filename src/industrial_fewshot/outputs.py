"""Readable result paths with atomic, non-overwriting repeat numbering."""
from pathlib import Path
import re


def safe_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', str(value)).strip(' ._') or 'model'
    if re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', name):
        name = '_' + name
    return name


def model_shot_name(config: dict) -> str:
    models = []
    for target in config['targets']:
        model = str(target['config']['model']).replace('\\', '/').rstrip('/').split('/')[-1]
        if target['backend'] in {'yoloe', 'yoloe_text', 'yoloe_fusion', 'groundingdino', 'yolo_onnx', 'pet_dino', 'wedetect'} and Path(model).suffix.lower() in {
                '.pt', '.pth', '.onnx', '.safetensors', '.bin'}:
            model = Path(model).stem
        if safe_name(model) not in models:
            models.append(safe_name(model))
    suite = config.get('experiment_suite', {})
    if len(config['targets']) == 1 and config['targets'][0]['backend'] == 'yoloe_fusion':
        settings = config['targets'][0]['config']['yoloe_fusion']
        return ('+'.join(models) + f"_{settings['positive_count']}shot_裁剪图{settings['visual_weight']:g}"
                + f"+文字{settings['text_weight']:g}")
    if all(target['backend'] == 'yolo_onnx' for target in config['targets']):
        return '+'.join(models) + '_监督检测'
    if suite.get('enabled'):
        counts = suite['positive_counts']
    else:
        counts = [max((len(target['config'].get('reference_examples', []))
                       or len(target['config'].get('reference_originals', []))
                       or len(target['config'].get('reference_boxed_images', []))
                       or len(target['config'].get('reference_crops', []))
                       or max(target['config'].get('max_reference_examples', 0),
                              target['config'].get('max_reference_crops', 0))
                       for target in config['targets']), default=0)]
    return '+'.join(models) + '_' + '+'.join(f'{count}shot' for count in counts)


def group_name(spec: dict, *, include_shot=False) -> str:
    if spec.get('positive_count') == 0:
        return '仅文字提示'
    names = {'original': '原图', 'crop': '裁剪图', 'boxed': '带框图', 'json': '标注JSON'}
    materials = '+'.join(names[part] for part in spec['positive_inputs'])
    negatives = spec.get('negative_count', 0)
    return (f"{spec['positive_count']}shot_" if include_shot else '') + materials + (
        f'_含{negatives}张负例' if negatives else '_无负例')


MODE_NAMES = {'text_only': '仅文字提示', 'text_json': '文字+标注JSON',
              'text_crops': '文字+目标裁剪', 'text_json_crops': '文字+标注JSON+目标裁剪'}


def result_directory(root: Path, name: str, *, create=False) -> Path:
    """Number siblings as 第2次, 第3次; never overwrite an earlier result."""
    root = Path(root)
    name = safe_name(name)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    repeat = 1
    while True:
        candidate = root / (name if repeat == 1 else f'{name}_第{repeat}次')
        if create:
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                pass
        elif not candidate.exists():
            return candidate
        repeat += 1
