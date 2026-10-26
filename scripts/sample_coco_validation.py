"""Sample a reproducible COCO validation subset from a tar archive, with LabelMe GT."""
from collections import Counter, defaultdict
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.labelme import build_labelme_document, load_ground_truth, read_json, write_json
from industrial_fewshot.paths import PROJECT_ROOT


def collect_hashes(value):
    if isinstance(value, dict):
        if value.get('image_sha256'):
            yield value['image_sha256']
        for child in value.values():
            yield from collect_hashes(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_hashes(child)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--support-manifests', type=Path, default=PROJECT_ROOT / 'data/references/diaowu')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('--count must be positive')
    source, output = args.archive.resolve(), args.output.resolve()
    staging = output.with_name(output.name + '.building')
    if output.exists() or staging.exists():
        raise ValueError(f'Output or staging already exists; choose a new output: {output}')
    source_stat = source.stat()
    references, support_sources = set(), []
    for path in sorted(args.support_manifests.glob('*/seed*.json')):
        hashes = set(collect_hashes(read_json(path)))
        references.update(hashes)
        support_sources.append({'path': str(path.resolve()), 'sha256': image_sha256(path), 'image_hash_count': len(hashes)})
    if not references:
        raise ValueError('No reference image hashes found in support manifests')
    archive_sha = image_sha256(source)
    print('Archive SHA256 verified; reference hashes:', len(references), flush=True)
    with tarfile.open(source, 'r:*') as archive:
        files = [member for member in archive if member.isfile()]
        by_name = {member.name: member for member in files}
        if len(by_name) != len(files):
            raise ValueError('Duplicate archive member names')
        annotations = [member for member in files if member.name.lower().endswith('.json')]
        if len(annotations) != 1:
            raise ValueError('Expected exactly one COCO annotation file')
        annotation_member = annotations[0]
        annotation_bytes = archive.extractfile(annotation_member).read()
        coco = json.loads(annotation_bytes.decode('utf-8-sig'))
        categories = {row['id']: row['name'] for row in coco['categories']}
        images = {row['id']: row for row in coco['images']}
        if len(images) != len(coco['images']) or len({im['file_name'].casefold() for im in images.values()}) != len(images):
            raise ValueError('Duplicate COCO image IDs or filenames')
        grouped_annotations = defaultdict(list)
        for annotation in coco['annotations']:
            if annotation['image_id'] not in images or annotation['category_id'] not in categories:
                raise ValueError('Unknown image/category in annotations')
            grouped_annotations[annotation['image_id']].append(annotation)
        root = PurePosixPath(annotation_member.name).parent.parent / 'images'
        buckets, invalid = defaultdict(list), []
        for image in sorted(images.values(), key=lambda row: row['file_name']):
            name = image['file_name']
            if PurePosixPath(name).name != name or '\\' in name or ':' in name:
                raise ValueError(f'Expected a flat safe image filename: {name}')
            member_name = str(root / name)
            if member_name not in by_name:
                raise ValueError(f'Missing image: {member_name}')
            problem = None
            for annotation in grouped_annotations[image['id']]:
                box = annotation.get('bbox')
                if (not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
                    problem = 'invalid_bbox'
                    break
                x, y, width, height = box
                if not (0 <= x < x + width <= image['width'] and 0 <= y < y + height <= image['height']):
                    problem = 'out_of_bounds_or_empty_bbox'
                if any(annotation.get(key) for key in ('iscrowd', 'ignore', 'difficult')):
                    problem = 'unsupported_ignore_or_crowd'
            if problem:
                invalid.append({'image_id': image['id'], 'file_name': name, 'reason': problem})
                continue
            group = Path(name).stem.rsplit('_', 1)[0]
            state = 'positive' if grouped_annotations[image['id']] else 'negative'
            buckets[group, state].append((image, member_name))
        total = sum(map(len, buckets.values()))
        if args.count > total:
            raise ValueError(f'Only {total} valid image/annotation pairs')
        quotas = {key: len(rows) * args.count // total for key, rows in buckets.items()}
        remainder = sorted(buckets, key=lambda key: (-(len(buckets[key]) * args.count % total), key))
        for key in remainder[:args.count - sum(quotas.values())]:
            quotas[key] += 1
        (staging / 'images').mkdir(parents=True)
        selected, rejected, used = [], [], set()
        rng = random.Random(args.seed)
        for key in sorted(buckets):
            candidates = list(buckets[key])
            rng.shuffle(candidates)
            accepted = 0
            for image, member_name in candidates:
                if accepted == quotas[key]:
                    break
                content = archive.extractfile(by_name[member_name]).read()
                checksum = hashlib.sha256(content).hexdigest()
                if checksum in references or checksum in used:
                    rejected.append({'image_id': image['id'], 'file_name': image['file_name'],
                                     'reason': 'reference_overlap' if checksum in references else 'duplicate_content',
                                     'image_sha256': checksum})
                    continue
                with Image.open(io.BytesIO(content)) as opened:
                    opened.load()
                    if opened.size != (image['width'], image['height']):
                        raise ValueError(f'COCO/image size mismatch: {image["file_name"]}')
                destination = staging / 'images' / image['file_name']
                destination.write_bytes(content)
                objects = []
                for annotation in grouped_annotations[image['id']]:
                    x, y, width, height = annotation['bbox']
                    objects.append({'label': categories[annotation['category_id']],
                                    'bbox_pixels_xyxy': [x, y, x + width, y + height]})
                document = build_labelme_document(image['file_name'], image['width'], image['height'], objects)
                for shape, annotation in zip(document['shapes'], grouped_annotations[image['id']]):
                    shape['source_annotation_id'] = annotation['id']
                label_path = destination.with_suffix('.json')
                write_json(label_path, document)
                truth = load_ground_truth(label_path, image['width'], image['height'], set(categories.values()))
                assert truth == objects
                selected.append({'relative_path': image['file_name'], 'source_image_id': image['id'],
                    'source_archive_member': member_name, 'sequence': key[0], 'presence': key[1],
                    'width': image['width'], 'height': image['height'], 'image_sha256': checksum,
                    'ground_truth': truth, 'ground_truth_source': 'json',
                    'annotation_sha256': image_sha256(label_path)})
                used.add(checksum)
                accepted += 1
            if accepted != quotas[key]:
                raise ValueError(f'Insufficient non-reference unique images in stratum {key}')
            print('SAMPLED', key, accepted, 'total', len(selected), flush=True)
    assert len(selected) == args.count and not (used & references)
    assert source.stat().st_size == source_stat.st_size and source.stat().st_mtime_ns == source_stat.st_mtime_ns
    selected.sort(key=lambda row: row['relative_path'])
    ids = {row['source_image_id'] for row in selected}
    subset = {**coco, 'images': [image for image in coco['images'] if image['id'] in ids],
              'annotations': [annotation for annotation in coco['annotations'] if annotation['image_id'] in ids]}
    write_json(staging / f'annotations/instances_val{args.count}.json', subset)
    manifest = {'schema': 'industrial_fewshot.split.v1', 'split': 'val',
                'image_root': str(output / 'images'), 'ground_truth_root': str(output / 'images'),
                'dataset_sha256': digest(selected), 'samples': selected, 'split_seed': args.seed,
                'group_by': 'sequence_and_target_presence', 'split_ratios': {'train': 0, 'val': 1, 'test': 0}}
    write_json(staging / 'manifest.json', manifest)
    old = PROJECT_ROOT / 'data/splits/diaowu/test.json'
    old_hashes = set(collect_hashes(read_json(old))) if old.is_file() else set()
    selection = {'schema': 'industrial_fewshot.coco_sample.v1', 'source_archive': str(source),
        'archive_format': 'tar (automatically detected; suffix may be misleading)',
        'source_archive_size': source_stat.st_size, 'source_archive_sha256': archive_sha,
        'source_annotation_member': annotation_member.name,
        'source_annotation_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
        'seed': args.seed, 'method': 'proportional stratified sampling by sequence and target presence; largest remainder quotas; sorted strata; Python Random.shuffle',
        'source_image_count': len(images), 'source_annotation_count': len(coco['annotations']),
        'sample_image_count': len(selected), 'sample_annotation_count': len(subset['annotations']),
        'presence_counts': dict(Counter(row['presence'] for row in selected)),
        'sequence_counts': dict(Counter(row['sequence'] for row in selected)),
        'support_manifests': support_sources, 'reference_image_hash_count': len(references),
        'reference_overlap_count': len(used & references), 'old_test_overlap_count': len(used & old_hashes),
        'excluded_invalid_images': invalid, 'rejected_candidates': rejected,
        'strata': [{'sequence': key[0], 'presence': key[1], 'available': len(buckets[key]), 'selected': quotas[key]} for key in sorted(buckets)],
        'samples': selected}
    write_json(staging / 'selection.json', selection)
    staging.rename(output)
    print(json.dumps({key: selection[key] for key in ('sample_image_count', 'sample_annotation_count', 'presence_counts', 'sequence_counts', 'reference_overlap_count', 'old_test_overlap_count')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
