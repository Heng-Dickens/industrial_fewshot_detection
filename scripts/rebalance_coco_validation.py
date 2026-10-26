"""Build a positive-rich, larger-target COCO subset without changing image/GT bytes."""
from collections import Counter, defaultdict
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT))
from PIL import Image
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.labelme import build_labelme_document, load_ground_truth, read_json, write_json
from scripts.sample_coco_validation import collect_hashes


def relative_areas(image,annotations):
    return [a['bbox'][2]*a['bbox'][3]/(image['width']*image['height']) for a in annotations]


def balanced_quotas(capacities,count):
    if count<0 or count>sum(capacities.values()):
        raise ValueError(f'Insufficient capacity: requested={count}, available={sum(capacities.values())}')
    result={key:0 for key in sorted(capacities)}
    while sum(result.values())<count:
        for key in result:
            if result[key]<capacities[key] and sum(result.values())<count:result[key]+=1
    return result


def spaced_selection(rows,count,seed):
    """Prefer images without tiny companion GT, then spread across available frames."""
    pending=list(rows);chosen=[]
    while len(chosen)<count:
        def score(row):
            distance=min((abs(row['frame']-old['frame']) for old in chosen),default=0)
            tie=hashlib.sha256(f"{seed}:{row['image']['file_name']}".encode()).hexdigest()
            return (row['all_boxes_large'],distance,tie)
        best=max(pending,key=score)
        chosen.append(best);pending.remove(best)
    return chosen


def statistics(coco,threshold):
    images={i['id']:i for i in coco['images']};grouped=defaultdict(list)
    for a in coco['annotations']:grouped[a['image_id']].append(a)
    ratios=[r for key,image in images.items() for r in relative_areas(image,grouped[key])]
    positive=sum(bool(grouped[key]) for key in images)
    return dict(image_count=len(images),positive_image_count=positive,negative_image_count=len(images)-positive,
        positive_fraction=positive/len(images),box_count=len(ratios),
        large_relative_area_threshold=threshold,
        images_with_large_target=sum(any(r>=threshold for r in relative_areas(image,grouped[key])) for key,image in images.items()),
        large_target_boxes=sum(r>=threshold for r in ratios),
        below_one_percent_boxes=sum(r<.01 for r in ratios),
        average_box_relative_area=sum(ratios)/len(ratios) if ratios else 0,
        sequence_counts=dict(Counter(Path(i['file_name']).stem.rsplit('_',1)[0] for i in images.values())))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/raw/diaowu_val300')
    p.add_argument('--output',type=Path,required=True,help='New staging directory; never overwrite a dataset')
    p.add_argument('--count',type=int,default=300)
    p.add_argument('--positive-count',type=int,default=210)
    p.add_argument('--large-area-ratio',type=float,default=.02)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--support-manifests',type=Path,default=ROOT/'data/references/diaowu')
    args=p.parse_args(argv)
    if not 0<args.positive_count<=args.count or not 0<args.large_area_ratio<1:
        p.error('Require 0 < positive-count <= count and 0 < large-area-ratio < 1')
    old=args.dataset.resolve();output=args.output.resolve()
    if output.exists():raise ValueError(f'Refusing to overwrite {output}')
    if not output.is_relative_to(ROOT/'data/raw') or output==old:
        raise ValueError('New output must be under project data/raw and differ from the active dataset')
    previous=read_json(old/'selection.json')
    baseline_files={name:old/name for name in ['selection.json','manifest.json','verification.json']}
    baseline_coco=old/'annotations/instances_val300.json';baseline_readme=old/'README.md'
    if (old/'history/v1_manifest.json').is_file():
        baseline_files={name:old/'history'/f'v1_{name}' for name in baseline_files}
        baseline_coco=old/'history/v1_instances_val300.json';baseline_readme=old/'history/v1_README.md'
    old_manifest=read_json(baseline_files['manifest.json']);old_coco=read_json(baseline_coco)
    source=Path(previous['source_archive']);source_stat=source.stat()
    source_sha=image_sha256(source)
    if source_sha!=previous['source_archive_sha256']:raise ValueError('Original archive hash changed')
    reference_hashes=set();support_records=[]
    for path in sorted(args.support_manifests.glob('*shot/seed*.json')):
        reference_hashes.update(collect_hashes(read_json(path)))
        support_records.append(dict(path=str(path.resolve()),sha256=image_sha256(path)))
    if not reference_hashes:raise ValueError('No verified support hashes found')
    frozen_references={str(f.relative_to(args.support_manifests)):image_sha256(f)
        for f in args.support_manifests.rglob('*') if f.is_file()}
    with tarfile.open(source,'r:*') as archive:
        members={m.name:m for m in archive if m.isfile()}
        annotation_name=previous['source_annotation_member']
        annotation_bytes=archive.extractfile(members[annotation_name]).read()
        if hashlib.sha256(annotation_bytes).hexdigest()!=previous['source_annotation_sha256']:
            raise ValueError('Source annotation bytes changed')
        coco=json.loads(annotation_bytes);images={i['id']:i for i in coco['images']}
        categories={c['id']:c['name'] for c in coco['categories']};grouped=defaultdict(list)
        for a in coco['annotations']:
            if a['image_id'] not in images or a['category_id'] not in categories:raise ValueError('Unknown annotation ID')
            grouped[a['image_id']].append(a)
        image_root=PurePosixPath(annotation_name).parent.parent/'images'
        buckets={'positive':defaultdict(list),'negative':defaultdict(list)}
        rejected=[];seen=set();available=Counter()
        for index,image in enumerate(sorted(images.values(),key=lambda i:i['file_name']),1):
            name=image['file_name']
            if PurePosixPath(name).name!=name or '\\' in name or ':' in name:raise ValueError('Unsafe image name')
            annotations=grouped[image['id']]
            for a in annotations:
                box=a['bbox']
                if (len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)
                        or not(0<=box[0]<box[0]+box[2]<=image['width'] and 0<=box[1]<box[1]+box[3]<=image['height'])
                        or any(a.get(k) for k in ['iscrowd','ignore','difficult'])):
                    raise ValueError(f'Unsupported source annotation: {a["id"]}')
            member=str(image_root/name)
            content=archive.extractfile(members[member]).read();checksum=hashlib.sha256(content).hexdigest()
            if checksum in reference_hashes or checksum in seen:
                rejected.append(dict(file_name=name,reason='reference_overlap' if checksum in reference_hashes else 'duplicate_content'))
                continue
            seen.add(checksum)
            ratios=relative_areas(image,annotations)
            if ratios and max(ratios)<args.large_area_ratio:
                available['smaller_only_positive']+=1
                continue
            presence='positive' if annotations else 'negative'
            sequence=Path(name).stem.rsplit('_',1)[0]
            row=dict(image=image,member=member,sha256=checksum,sequence=sequence,
                     frame=int(Path(name).stem.rsplit('_',1)[1]),
                     all_boxes_large=bool(ratios) and min(ratios)>=args.large_area_ratio)
            buckets[presence][sequence].append(row);available[presence]+=1
            if index%200==0:print('INDEXED',index,len(images),flush=True)
        desired={'positive':args.positive_count,'negative':args.count-args.positive_count}
        chosen=[];quota_records={}
        for presence,count in desired.items():
            quotas=balanced_quotas({key:len(rows) for key,rows in buckets[presence].items()},count)
            quota_records[presence]=quotas
            for sequence,quota in quotas.items():chosen.extend(spaced_selection(buckets[presence][sequence],quota,args.seed))
        chosen.sort(key=lambda r:r['image']['file_name'])
        output.mkdir(parents=True);(output/'images').mkdir()
        selected=[]
        for row in chosen:
            image=row['image'];name=image['file_name'];content=archive.extractfile(members[row['member']]).read()
            assert hashlib.sha256(content).hexdigest()==row['sha256']
            with Image.open(io.BytesIO(content)) as opened:
                opened.load();assert opened.size==(image['width'],image['height'])
            path=output/'images'/name;path.write_bytes(content)
            truth=[]
            for a in grouped[image['id']]:
                x,y,w,h=a['bbox'];truth.append(dict(label=categories[a['category_id']],bbox_pixels_xyxy=[x,y,x+w,y+h]))
            document=build_labelme_document(name,image['width'],image['height'],truth)
            for shape,a in zip(document['shapes'],grouped[image['id']]):shape['source_annotation_id']=a['id']
            write_json(path.with_suffix('.json'),document)
            assert load_ground_truth(path.with_suffix('.json'),image['width'],image['height'],set(categories.values()))==truth
            selected.append(dict(relative_path=name,source_image_id=image['id'],source_archive_member=row['member'],
                sequence=row['sequence'],presence='positive' if truth else 'negative',width=image['width'],height=image['height'],
                image_sha256=row['sha256'],annotation_sha256=image_sha256(path.with_suffix('.json')),
                ground_truth=truth,ground_truth_source='json'))
    ids={r['source_image_id'] for r in selected}
    subset={**coco,'images':[i for i in coco['images'] if i['id'] in ids],
            'annotations':[a for a in coco['annotations'] if a['image_id'] in ids]}
    before=statistics(old_coco,args.large_area_ratio);after=statistics(subset,args.large_area_ratio)
    assert len(selected)==args.count and after['positive_image_count']==args.positive_count
    assert after['images_with_large_target']==args.positive_count
    assert after['large_target_boxes']/after['box_count']>before['large_target_boxes']/before['box_count']
    assert {r['image_sha256'] for r in selected}.isdisjoint(reference_hashes)
    assert len({r['image_sha256'] for r in selected})==args.count
    assert frozen_references=={str(f.relative_to(args.support_manifests)):image_sha256(f)
        for f in args.support_manifests.rglob('*') if f.is_file()}
    assert (source.stat().st_size,source.stat().st_mtime_ns)==(source_stat.st_size,source_stat.st_mtime_ns)
    revision=f'large_targets_positive{round(args.positive_count*100/args.count)}_v2';fingerprint=digest(selected)
    write_json(output/'annotations/instances_val300.json',subset)
    manifest=dict(schema='industrial_fewshot.split.v1',split='val',image_root=str(old/'images'),
        ground_truth_root=str(old/'images'),dataset_sha256=fingerprint,samples=selected,split_seed=args.seed,
        group_by='sequence_target_presence_relative_area',split_ratios={'train':0,'val':1,'test':0},
        dataset_revision=revision,previous_dataset_sha256=old_manifest['dataset_sha256'])
    write_json(output/'manifest.json',manifest)
    record=dict(schema='industrial_fewshot.coco_sample.v2',source_archive=str(source),source_archive_sha256=source_sha,
        source_archive_size=source_stat.st_size,source_annotation_member=annotation_name,
        source_annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),seed=args.seed,
        method='all positives contain >=threshold target; prefer all-box-large; equal sequence quotas with capacity limits; farthest-frame selection',
        dataset_revision=revision,previous_dataset_sha256=old_manifest['dataset_sha256'],dataset_sha256=fingerprint,
        sample_image_count=args.count,sample_annotation_count=len(subset['annotations']),support_manifests=support_records,
        reference_image_hash_count=len(reference_hashes),reference_overlap_count=0,source_available_counts=dict(available),
        quotas=quota_records,rejected_candidates=rejected,before=before,after=after,samples=selected,
        old_cohort_overlap_count=len(ids & {i['id'] for i in old_coco['images']}))
    write_json(output/'selection.json',record)
    for name in ['selection.json','manifest.json','verification.json']:
        write_json(output/'history'/f"v1_{name}",read_json(baseline_files[name]))
    write_json(output/'history/v1_instances_val300.json',old_coco)
    (output/'history/v1_README.md').write_bytes(baseline_readme.read_bytes())
    write_json(output/'verification.json',dict(status='built_and_verified',before=before,after=after,
        dataset_sha256=fingerprint,source_archive_sha256=source_sha,all_selected_images_decoded=True,
        image_bytes_match_source=True,all_original_annotations_preserved=True,reference_overlap_count=0,
        reference_files_unchanged=True,project_dataset_loader_passed=False))
    print(json.dumps(dict(output=str(output),before=before,after=after,dataset_sha256=fingerprint),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
