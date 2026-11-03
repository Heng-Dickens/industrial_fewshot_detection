"""Generate fresh frozen-support YOLOE visual candidates for dev and test."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.detectors.yoloe_visual import YOLOERunner, prepare_references


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def folder(scene, split, stamp):
    return ROOT / 'outputs/OVD-MLLM/experiments' / f'{scene}_{split}' / stamp


def generate(scene, device, stamp):
    labels = ['Chache1', 'Chache2', 'Chache3', 'Chache4', 'Qianyinche'] if scene == 'chache' else ['hanging object']
    manifests = {split: read(ROOT / f'data/raw/{scene}_{split}/manifest.json') for split in ('val100', 'val300')}
    hashes = [{s['image_sha256'] for s in d['samples']} for d in manifests.values()]
    if hashes[0] & hashes[1]:
        raise ValueError('Development/test image content overlap')
    sequences = [{s.get('sequence') for s in d['samples']} for d in manifests.values()]
    audit = dict(exact_image_overlap=0, shared_sequences=sorted(sequences[0] & sequences[1]),
                 limitation='Existing cohorts may share video sequences; no claim of sequence-independent generalization.')
    all_references = []
    for label in labels:
        task = label if scene == 'chache' else 'diaowu'
        path = ROOT / f'outputs/_shared/runtime/aligned_baselines_20261027/configs/{scene}_val300_yoloe_10shot_{task}.local.json'
        config = copy.deepcopy(load_config(path)['targets'][0]['config'])
        config['yoloe'].update(device=str(device), confidence=.001, nms_iou=.7, max_detections=100)
        prepare_references(config)
        references = config['reference_examples']
        if any(r['image_sha256'] in hashes[0] | hashes[1] for r in references):
            raise ValueError('Support/query content overlap')
        all_references.extend(dict(r, label=label) for r in references)
        identity = dict(schema='ovd_mllm.yoloe.v1', scene=scene, label=label, positive_count=10,
                        weights_sha256=sha(config['yoloe']['weights']),
                        adapter_sha256=sha(ROOT / 'src/industrial_fewshot/detectors/yoloe_visual.py'),
                        references=references, settings={k:v for k,v in config['yoloe'].items() if k!='device'})
        pending = []
        for split, manifest in manifests.items():
            target = folder(scene, split, stamp) / 'source' / task
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'config.json').exists() and read(target / 'config.json') != identity:
                raise ValueError('Source config changed: use a new stamp')
            write(target / 'config.json', identity)
            records_path = target / 'records.jsonl'
            latest = {}
            if records_path.exists():
                for line in records_path.read_text().splitlines():
                    r = json.loads(line)
                    latest[r['relative_path']] = r
            for sample in manifest['samples']:
                old = latest.get(sample['relative_path'])
                if old and old.get('status') == 'success':
                    if old['image_sha256'] != sample['image_sha256']:
                        raise ValueError('Image changed')
                    continue
                pending.append((split, sample, records_path))
        if not pending:
            print(scene, label, 'already complete', flush=True)
            continue
        runner = YOLOERunner(config)
        for number, (split, sample, records_path) in enumerate(pending, 1):
            image = ROOT / f'data/raw/{scene}_{split}/images' / sample['relative_path']
            data = image.read_bytes()
            if hashlib.sha256(data).hexdigest() != sample['image_sha256']:
                raise ValueError('Image content changed: ' + str(image))
            started = time.perf_counter()
            prediction = runner.predict(data)
            record = dict(relative_path=sample['relative_path'], image_sha256=sample['image_sha256'],
                          target_id=task, status='success', inference_seconds=time.perf_counter()-started,
                          **prediction)
            with records_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            if number % 25 == 0 or number == len(pending):
                print(scene, label, number, '/', len(pending), flush=True)
        del runner
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    for split, manifest in manifests.items():
        output = folder(scene, split, stamp)
        write(output / 'cohort.json', dict(manifest, image_root=str(ROOT / f'data/raw/{scene}_{split}/images')))
        write(output / 'references.json', {'examples':all_references})
        write(output / 'split_audit.json', audit)
        merged = []
        by_label = {}
        for label in labels:
            task = label if scene == 'chache' else 'diaowu'
            records = [json.loads(s) for s in (output / 'source' / task / 'records.jsonl').read_text().splitlines()]
            by_label[label] = {r['relative_path']:r for r in records}
        for sample in manifest['samples']:
            objects = []
            for label in labels:
                record = by_label[label][sample['relative_path']]
                start = len(objects)
                objects.extend(dict(obj, candidate_id=f'p{start+i:04}') for i,obj in enumerate(record['objects']))
            merged.append(dict(relative_path=sample['relative_path'], image_sha256=sample['image_sha256'],
                               target_id='all', status='success', objects=objects))
        (output / 'source_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
        write(output / 'source_complete.json', dict(scene=scene, split=split, images=len(merged),
                predictions=sum(len(r['objects']) for r in merged), candidate_floor=.001,
                reference_shot_per_class=10, labels=labels, stamp=stamp))
        print('WROTE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['chache','diaowu'], required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--stamp', default='yoloe_v8l_10shot_qwen38_v1_20261030')
    args = parser.parse_args()
    generate(args.scene, args.device, args.stamp)
