"""Run paired 5-shot YOLOv8/YOLOE comparisons using the public training/evaluation entrypoints."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STAMP = 'yolo_5shot_20261102'
RUNTIME = ROOT / 'outputs/_shared/runtime/yolo_5shot_comparison_20261102'
COMMON = ROOT / 'outputs/YOLOE/finetune/scripts/common'
JOBS = [dict(scene=scene, method=method, gpu=gpu, id=f'{scene}_{method}')
        for gpu, (scene, method) in enumerate([
            ('chache', 'yolov8'), ('chache', 'yoloe-head'), ('chache', 'yoloe-vp'),
            ('diaowu', 'yolov8'), ('diaowu', 'yoloe-head'), ('diaowu', 'yoloe-vp'),
            ('chache', 'yolov8-untrained'), ('diaowu', 'yolov8-untrained')])]
LABELS = {'yolov8': 'YOLOv8-L 5shot微调', 'yoloe-head': 'YOLOE固定头 5shot微调',
          'yoloe-vp': 'YOLOE保留并训练SAVPE 5shot微调',
          'yolov8-untrained': 'YOLOv8-L改目标分类输出、不训练', 'yoloe-original': '原始YOLOE视觉提示'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locations(job):
    cfg = RUNTIME / 'configs' / f'{job["scene"]}_{"vp" if job["method"] == "yoloe-vp" else "fixed"}.json'
    run = ROOT / read(cfg)['project'] / job['method']
    return cfg, run


def commands(job, execute=True):
    cfg, run = locations(job)
    baseline = job['method'] == 'yolov8-untrained'
    train = [sys.executable, str(COMMON/'train.py'), '--config', str(cfg),
             '--method', 'yolov8' if baseline else job['method'], '--name', run.name,
             '--device', '0', '--batch', '4']
    if job['method'] == 'yoloe-head':
        train += ['--strategy', 'full']
    if baseline:
        train += ['--initialize-only']
    if execute:
        train += ['--execute']
    evaluations = []
    for split in ['dev', 'test']:
        cmd = [sys.executable, str(COMMON/'evaluate.py'), '--config', str(cfg),
               '--run-dir', str(run), '--split', split, '--device', '0']
        if baseline:
            cmd += ['--checkpoint', 'initialized']
        if job['method'] == 'yoloe-vp':
            cmd += ['--prompt-shots', '5']
        if execute:
            cmd += ['--execute']
        evaluations.append(cmd)
    return train, evaluations


def worker(job):
    statefile = RUNTIME / 'jobs' / (job['id'] + '.json')
    cfg, run = locations(job)
    state = dict(job, pid=os.getpid(), started=time.time(), config=str(cfg), run_dir=str(run),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), status='running')
    write(statefile, state)
    try:
        train, evaluations = commands(job)
        meta = run/'finetune_config.json'
        expected = 'initialized_no_training' if job['method'] == 'yolov8-untrained' else 'completed'
        if not meta.exists() or read(meta)['status'] != expected:
            state.update(stage='initialization' if job['method'] == 'yolov8-untrained' else 'training', argv=train)
            write(statefile, state)
            print(json.dumps(dict(argv=train), ensure_ascii=False), flush=True)
            subprocess.run(train, cwd=ROOT, check=True)
        for split, cmd in zip(['dev', 'test'], evaluations):
            output = run / ('dev_val100' if split == 'dev' else 'test_val300')
            mode = 'visual_5shot' if job['method'] == 'yoloe-vp' else 'fixed_head'
            complete = output/mode/'run.json'
            if complete.exists() and read(complete).get('status') == 'completed':
                continue
            state.update(stage=f'evaluate_{split}', argv=cmd)
            write(statefile, state)
            print(json.dumps(dict(argv=cmd), ensure_ascii=False), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        state.update(status='completed', stage='complete', finished=time.time())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished=time.time())
        traceback.print_exc()
        raise
    finally:
        write(statefile, state)


def launch():
    for job in JOBS:
        statefile = RUNTIME/'jobs'/(job['id']+'.json')
        if statefile.exists():
            state = read(statefile)
            if state['status'] == 'completed':
                continue
            if state['status'] == 'running':
                try:
                    os.kill(state['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'Already running: {job["id"]}')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   YOLO_AUTOINSTALL='false')
        log = RUNTIME/'logs'/(job['id']+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', job['id']],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(job['id'], 'physical GPU', job['gpu'], 'pid', process.pid, 'log', log.relative_to(ROOT), flush=True)


def result_row(scene, method, source):
    metrics = read(source/'metrics.json')
    run = read(source/'run.json')
    if metrics.get('evaluation_status') != 'complete' or metrics.get('evaluated_image_count') != 300:
        raise ValueError(f'Incomplete test metrics: {source}')
    return dict(scene=scene, method=method, method_display=LABELS[method],
                training_shot_per_class=5 if method not in {'yolov8-untrained', 'yoloe-original'} else 0,
                inference_shot_per_class=5 if method in {'yoloe-vp', 'yoloe-original'} else None,
                **{k:metrics[k] for k in ['mAP50','mAP50_95','precision','recall','f2','tp','fp','fn']},
                source=str(source.relative_to(ROOT)), metrics_sha256=sha(source/'metrics.json'),
                dataset_sha256=run.get('dataset_sha256'), weights_sha256=run.get('weights_sha256'))


def preview(scene, source):
    """Render saved predictions on a deterministic positive test image; no inference."""
    from PIL import Image, ImageDraw, ImageFont
    manifest = read(ROOT/f'data/raw/{scene}_val300/manifest.json')
    sample = next(s for s in manifest['samples'] if s['ground_truth'])
    records = [json.loads(line) for line in (source/'records.jsonl').read_text().splitlines() if line.strip()]
    records = [r for r in records if r['relative_path'] == sample['relative_path']]
    image = Image.open(ROOT/f'data/raw/{scene}_val300/images'/sample['relative_path']).convert('RGB')
    width, height = image.size
    image.thumbnail((1600, 1000))
    sx, sy = image.width/width, image.height/height
    draw = ImageDraw.Draw(image)
    font_path = Path.home()/'.config/Ultralytics/Arial.ttf'
    font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    for objects, color, prefix in [(sample['ground_truth'], '#00cc44', 'GT'),
            ([p for r in records for p in r.get('objects', []) if p['confidence'] >= .5], '#ff4444', 'Pred')]:
        for obj in objects:
            x1,y1,x2,y2=obj['bbox_pixels_xyxy']
            box=(x1*sx,y1*sy,x2*sx,y2*sy)
            draw.rectangle(box,outline=color,width=3)
            text=prefix+' '+obj['label']+(f' {obj["confidence"]:.2f}' if prefix=='Pred' else '')
            draw.text((box[0],max(25,box[1]-22)),text,fill=color,font=font)
    draw.rectangle((0,0,image.width,25), fill='black')
    draw.text((5,2),'Green: GT | Red: predictions score >= 0.5 | '+sample['relative_path'],fill='white',font=font)
    image.save(source/'preview.jpg',quality=90)


def verify_checkpoint(job):
    """Verify actual backbone/SAVPE updates in each selected trained checkpoint."""
    import torch
    sys.path.insert(0, str(COMMON))
    sys.path.insert(0, str(ROOT/'third_party/yoloe'))
    from ultralytics import YOLO, YOLOE
    torch.set_num_threads(2)
    cfg, run = locations(job)
    metadata=read(run/'finetune_config.json')
    if job['method']=='yolov8-untrained':
        assert metadata['optimization_steps']==0
        return metadata['initialization_verification']
    loader=YOLO if job['method']=='yolov8' else YOLOE
    original=loader(metadata['paths']['weights']).model.float().eval()
    trained=loader(str(run/'weights/best.pt')).model.float().eval()
    changed=not torch.equal(original.model[0].conv.weight,trained.model[0].conv.weight)
    if not changed:
        raise RuntimeError(f'Trained backbone unchanged: {run}')
    result=dict(backbone_weights_changed=True,weights_sha256=sha(run/'weights/best.pt'))
    head=trained.model[-1]
    if job['method']=='yoloe-vp':
        assert hasattr(head,'savpe') and not getattr(head,'is_fused',False)
        before=dict(original.model[-1].savpe.named_parameters())
        changed_parameters=[name for name,p in head.savpe.named_parameters()
                            if not torch.equal(p,before[name])]
        if not changed_parameters:
            raise RuntimeError(f'SAVPE parameters unchanged: {run}')
        result.update(savpe_present=True,savpe_weights_changed=True,
                      savpe_changed_parameter_count=len(changed_parameters))
    else:
        result['savpe_present']=hasattr(head,'savpe')
        if job['method']=='yoloe-head':
            assert not result['savpe_present']
    write(run/'comparison_verification.json',result)
    return result


def summarize():
    states = [read(RUNTIME/'jobs'/(j['id']+'.json')) for j in JOBS]
    if not all(s['status'] == 'completed' for s in states):
        raise RuntimeError('Some jobs have not completed; no partial ranking written.')
    all_rows = []
    verifications={j['id']:verify_checkpoint(j) for j in JOBS}
    for scene in ['chache', 'diaowu']:
        rows = []
        for method in ['yolov8-untrained', 'yolov8', 'yoloe-head', 'yoloe-vp']:
            job = next(j for j in JOBS if j['scene']==scene and j['method']==method)
            cfg, run = locations(job)
            mode = 'visual_5shot' if method == 'yoloe-vp' else 'fixed_head'
            source=run/'test_val300'/mode
            rows.append(result_row(scene, method, source))
            preview(scene, source)
        original = ROOT/f'outputs/YOLOE/experiments/{scene}_val300/yoloe-v8l-seg_5shot'
        rows.append(result_row(scene, 'yoloe-original', original))
        all_rows.extend(rows)
        destination = ROOT/f'outputs/YOLOE/finetune/{scene}_val300/comparisons/{STAMP}'
        destination.mkdir(parents=True, exist_ok=True)
        with (destination/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(destination/'results.json',dict(protocol=protocol(), results=rows,
              checkpoint_verification={k:v for k,v in verifications.items() if k.startswith(scene+'_')}))
        render(destination/'README.md', rows, scene)
        index=destination.parent/'README.md'
        entry=f'[5shot训练与完整测试推理对照]({STAMP}/README.md)'
        text=index.read_text() if index.exists() else f'# {scene}微调比较\n'
        if entry not in text:
            index.write_text(text.rstrip()+'\n\n'+entry+'\n')
    render(ROOT/'outputs/YOLOE/finetune/5SHOT_COMPARISON.md', all_rows, '两个场景')


def protocol():
    return dict(training_shot=5, support_seed=0, training_seed=0, imgsz=640, batch=4,
                epochs=100, patience=10, optimizer='AdamW', lr0=0.0001, fp32=True,
                dev_images=100, test_images=300, confidence_floor=0.001, nms_iou=0.7,
                metric_iou=0.5, operating_score=0.5, max_detections_per_class=100,
                vp=dict(scope='full', protocol='same_image', savpe_trainable=True,
                        savpe_bn_running_stats_frozen=True, inference_shot=5))


def render(path, rows, title):
    lines=[f'# {title}：YOLOv8与YOLOE的5shot训练及推理对照','',
        '训练集：叉车每类5张、共25张；吊物5张。两任务分别使用原100张dev选best、300张test评估。',
        '训练seed与参考集seed均为0；640输入、FP32、batch=4、AdamW lr=1e-4、最多100epoch、patience=10。三种微调均更新backbone；SAVPE方案使用same_image全量训练，固定SAVPE的BN运行统计，权重仍参与训练。',
        '未训练YOLOv8使用目标类别数重新构造检测模型，迁移全部形状兼容的预训练张量，仅目标分类输出随机初始化；优化步数为0。它是随机目标分类输出的负对照，不能解释为YOLOv8的语义零样本能力。原始YOLOE一行复用已保存的同一300张、seed=0、5shot视觉提示结果。','',
        '| 场景 | 方法 | 训练shot/类 | 推理参考shot/类 | AP50 % | AP50:95 % | P@0.5 % | R@0.5 % | F2@0.5 % | TP/FP/FN | 详情 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        detail=ROOT/row['source']/'evaluation.txt'
        link=os.path.relpath(detail,path.parent)
        metrics=' | '.join(f'{row[k]*100:.2f}' for k in ['mAP50','mAP50_95','precision','recall','f2'])
        counts='/'.join(str(row[k]) for k in ['tp','fp','fn'])
        preview_link=os.path.relpath(detail.with_name('preview.jpg'),path.parent)
        extra=f' · [示例]({preview_link})' if detail.with_name('preview.jpg').exists() else ''
        lines.append(f'| {row["scene"]} | {row["method_display"]} | {row["training_shot_per_class"]} | {row["inference_shot_per_class"] or "不适用"} | {metrics} | {counts} | [评估]({link}){extra} |')
    lines+=['','所有AP来自保存的完整候选；P/R/F2为score≥0.5、IoU≥0.5。固定头联合多类查询；SAVPE逐类查询后合并，查询与NMS竞争方式不同。每个结果均保留逐图预测、原始指标与权重身份；本轮为单训练seed实验。','']
    path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=[j['id'] for j in JOBS])
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(next(j for j in JOBS if j['id']==args.worker))
    elif args.launch:
        launch()
    elif args.summarize:
        summarize()
    else:
        for job in JOBS:
            p=RUNTIME/'jobs'/(job['id']+'.json')
            state=read(p) if p.exists() else dict(status='not_started')
            print(job['id'],state.get('status'),state.get('stage'),state.get('error',''))


if __name__=='__main__':
    main()
