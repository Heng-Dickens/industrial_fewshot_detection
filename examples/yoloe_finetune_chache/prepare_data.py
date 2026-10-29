"""Convert frozen support samples and the selected scene's 100-image LabelMe dev set.

Default: inspect/print a plan. Only --write copies images and creates YOLO labels.
Never resplit or modify the existing 300-image evaluation dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

try:
    from .finetune_config import DEFAULT_CONFIG, load_config
except ImportError:
    from finetune_config import DEFAULT_CONFIG, load_config

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['Chache1', 'Chache2', 'Chache3', 'Chache4', 'Qianyinche']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def sample(image, annotation, classes=CLASSES):
    # Read actual pixels; a missing annotation is an error, never an implicit negative.
    from PIL import Image

    document = read(annotation)
    with Image.open(image) as opened:
        width, height = opened.size
        opened.verify()
    if (document.get('imageWidth'), document.get('imageHeight')) != (width, height):
        raise ValueError(f'Annotation/image dimensions differ: {annotation}')
    rows, counts = [], Counter()
    for shape in document['shapes']:
        label = shape['label']
        if label not in classes:
            raise ValueError(f'Unknown class {label!r}: {annotation}')
        points = shape['points']
        if shape.get('shape_type') not in {'rectangle', 'polygon'} or len(points) < 2:
            raise ValueError(f'Expected rectangle or polygon: {annotation}')
        if any(len(p) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                 for v in p) for p in points):
            raise ValueError(f'Invalid coordinates: {annotation}')
        x1, x2 = min(p[0] for p in points), max(p[0] for p in points)
        y1, y2 = min(p[1] for p in points), max(p[1] for p in points)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Empty/out-of-bounds box: {annotation}')
        box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
               (x2 - x1) / width, (y2 - y1) / height)
        rows.append(str(classes.index(label)) + ' ' + ' '.join(f'{v:.9f}' for v in box))
        counts[label] += 1
    return dict(image=str(image), annotation=str(annotation), image_sha256=sha256(image),
                annotation_sha256=sha256(annotation), labels=rows, box_counts=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--shot', type=int, choices=[1, 5, 10], help='Defaults to the scene configuration.')
    parser.add_argument('--support-seed', type=int, help='Defaults to the scene configuration.')
    parser.add_argument('--dev-dir', type=Path,
                        help='Independent dev images and same-stem LabelMe JSONs, recursively searched.')
    parser.add_argument('--dev-manifest', type=Path, help='Optional override of the frozen dev image list.')
    parser.add_argument('--output', type=Path, help='New output directory; refuses existing paths.')
    parser.add_argument('--write', action='store_true', help='Actually write the prepared dataset.')
    args = parser.parse_args()
    config = load_config(args.config)
    scene, classes = config['scene'], config['class_names']
    args.shot = args.shot if args.shot is not None else config['shot']
    args.support_seed = args.support_seed if args.support_seed is not None else config['support_seed']
    default_output = ROOT / config['data']
    default_output = default_output.parent if args.shot == config['shot'] else ROOT / f'data/finetune/{scene}_{args.shot}shot_val100'
    output = (args.output or default_output).resolve()

    # Step 1: read the exact same frozen support samples as the prompting baseline.
    train, support_sources = [], []
    for label in classes:
        reference = ROOT / 'data/references' / scene
        if scene == 'chache':
            reference = reference / label
        frozen_path = reference / f'{args.shot}shot/seed_{args.support_seed}.json'
        frozen = read(frozen_path)
        if len(frozen['positive_samples']) != args.shot:
            raise ValueError(f'Unexpected support count: {frozen_path}')
        index = {r['image']: r for r in read(reference / 'manifest.json')['images']}
        support_sources.append(dict(path=str(frozen_path), sha256=sha256(frozen_path)))
        for chosen in frozen['positive_samples']:
            original = index[chosen['sample_id']]
            row = sample(reference / original['image'], reference / original['annotation'], classes)
            if row['image_sha256'] != chosen['image_sha256'] or row['annotation_sha256'] != chosen['annotation_sha256']:
                raise ValueError(f'Frozen support bytes changed: {chosen["sample_id"]}')
            if not row['box_counts'].get(label):
                raise ValueError(f'Support lacks its declared class {label}: {row["annotation"]}')
            row['support_class'] = label
            train.append(row)

    # Step 2: dev images must come from an independently prepared, fully annotated set.
    configured_dev = (ROOT / config['dev_images']).resolve()
    dev_dir = args.dev_dir.resolve() if args.dev_dir else configured_dev
    if not dev_dir.is_dir():
        raise ValueError(f'Dev directory does not exist: {dev_dir}')
    dev_manifest_path = args.dev_manifest.resolve() if args.dev_manifest else (
        (ROOT / config['dev_manifest']).resolve() if dev_dir == configured_dev else None)
    dev_manifest = read(dev_manifest_path) if dev_manifest_path else None
    if dev_manifest:
        candidates = [dev_dir / r['relative_path'] for r in dev_manifest['samples']]
    else:
        candidates = [p for p in sorted(dev_dir.rglob('*')) if p.is_file()
                      and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    dev = [sample(p, p.with_suffix('.json'), classes) for p in candidates]
    if not dev:
        raise ValueError(f'No supported dev images found in {dev_dir}; provide original images and same-stem LabelMe JSONs.')
    if dev_dir == configured_dev and len(dev) != config['dev_expected_images']:
        raise ValueError(f'Expected {config["dev_expected_images"]} frozen dev images, got {len(dev)}: {dev_dir}')
    if dev_manifest:
        for row, frozen in zip(dev, dev_manifest['samples']):
            if row['image_sha256'] != frozen['image_sha256'] or row['annotation_sha256'] != frozen['annotation_sha256']:
                raise ValueError(f'Dev bytes differ from the frozen manifest: {row["image"]}')
    dev_classes = Counter(k for row in dev for k in row['box_counts'])
    if set(dev_classes) != set(classes):
        raise ValueError(f'Dev set must cover {classes}, currently: {dict(dev_classes)}')

    # Step 3: use ONLY evaluation-image hashes for exclusion, never evaluation GT for training.
    benchmark_path = ROOT / config['test_manifest']
    benchmark = read(benchmark_path)
    test_hashes = {s['image_sha256'] for s in benchmark['samples']}
    seen = set()
    for split, rows in [('train', train), ('val', dev)]:
        for row in rows:
            digest = row['image_sha256']
            if digest in seen or digest in test_hashes:
                raise ValueError(f'Duplicate or train/dev/test overlap: {row["image"]}')
            seen.add(digest)
            row['split'] = split
            # Content hash avoids same-name collisions between cameras or classes.
            row['output_stem'] = digest

    plan = dict(schema='industrial_fewshot.finetune_data.v2', scene=scene, shot_per_class=args.shot,
                support_seed=args.support_seed, classes=classes, support_manifests=support_sources,
                dev_manifest=str(dev_manifest_path) if dev_manifest_path else None,
                dev_manifest_sha256=sha256(dev_manifest_path) if dev_manifest_path else None,
                dev_dataset_sha256=dev_manifest.get('dataset_sha256') if dev_manifest else None,
                excluded_test_manifest=str(benchmark_path), excluded_test_manifest_sha256=sha256(benchmark_path),
                splits={split: dict(image_count=len(rows),
                        negative_images=sum(not r['labels'] for r in rows),
                        boxes=dict(sum((Counter(r['box_counts']) for r in rows), Counter())))
                        for split, rows in [('train', train), ('val', dev)]},
                samples=train + dev)
    print(json.dumps(dict(scene=scene, output=str(output), shot_per_class=args.shot, splits=plan['splits'],
                         write=args.write), ensure_ascii=False, indent=2))
    if not args.write:
        print('Plan only. Review annotations and camera/video separation before using --write.')
        return
    if output.exists():
        raise FileExistsError(f'Choose a new output path; refusing to overwrite: {output}')

    # Step 4: copy originals, write normalized detection labels, then write canonical lists/YAML.
    output.mkdir(parents=True)
    for split, rows in [('train', train), ('val', dev)]:
        images, labels = output / 'images' / split, output / 'labels' / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        paths = []
        for row in rows:
            source = Path(row['image'])
            destination = images / (row['output_stem'] + source.suffix.lower())
            shutil.copy2(source, destination)
            (labels / (row['output_stem'] + '.txt')).write_text(
                '\n'.join(row['labels']) + ('\n' if row['labels'] else ''), encoding='utf-8')
            paths.append(destination.as_posix())
        (output / f'{split}.txt').write_text('\n'.join(paths) + '\n', encoding='utf-8')
    # JSON strings are valid YAML scalars and safely quote spaces or non-ASCII paths.
    data_yaml = '\n'.join(['path: ' + json.dumps(output.as_posix()), 'train: train.txt', 'val: val.txt',
                           'names:', *[f'  {i}: {json.dumps(name)}' for i, name in enumerate(classes)]]) + '\n'
    (output / 'dataset.yaml').write_text(data_yaml, encoding='utf-8')
    (output / 'preparation.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared dataset: {output / "dataset.yaml"}')


if __name__ == '__main__':
    main()
