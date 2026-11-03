"""Fresh FP32 ONNX evaluation of both custom JSTY detectors on frozen 300-image cohorts."""
from pathlib import Path
import argparse
import ast
from collections import Counter
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
CLASSES={'chache':['Chache1','Chache2','Chache3','Chache4','Qianyinche'],
         'diaowu':['hanging object']}
WEIGHTS={'chache':'jsty_special_vehicle/jsty_chache_det5_960_1280_0_0_0_2_fp32.onnx',
         'diaowu':'jsty_hanging_object/jsty_hanging_object_det1_0_0_0_1_fp32.onnx'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def cohort(scene):
    from PIL import Image
    from industrial_fewshot.data.labelme import load_evaluation_truth
    manifest=ROOT/f'data/raw/{scene}_val300/manifest.json'
    data=read(manifest)
    items=[]
    for sample in sorted(data['samples'],key=lambda r:r['relative_path']):
        image=manifest.parent/'images'/sample['relative_path']
        annotation=image.with_suffix('.json')
        assert sha(image)==sample['image_sha256'],image
        assert sha(annotation)==sample['annotation_sha256'],annotation
        with Image.open(image) as opened:
            width,height=opened.size
            opened.verify()
        assert (width,height)==(sample['width'],sample['height']),image
        truth,origin=load_evaluation_truth(annotation,width,height,set(CLASSES[scene]),'error')
        assert truth==sample['ground_truth'],annotation
        items.append(dict(path=str(image),relative_path=sample['relative_path'],width=width,height=height,
                          image_sha256=sample['image_sha256'],ground_truth=truth,ground_truth_source=origin))
    assert len(items)==300 and len({i['image_sha256'] for i in items})==300
    return manifest,data,items


def preview(source,item,objects):
    from PIL import Image,ImageDraw,ImageFont
    image=Image.open(item['path']).convert('RGB');width,height=image.size
    image.thumbnail((1600,1000));sx,sy=image.width/width,image.height/height
    draw=ImageDraw.Draw(image)
    fontpath=Path.home()/'.config/Ultralytics/Arial.ttf'
    font=ImageFont.truetype(str(fontpath),17) if fontpath.is_file() else ImageFont.load_default()
    for rows,color,prefix in [(item['ground_truth'],'#00cc44','GT'),
            ([o for o in objects if o['confidence']>=.5],'#ff4444','Pred')]:
        for obj in rows:
            x1,y1,x2,y2=obj['bbox_pixels_xyxy'];box=(x1*sx,y1*sy,x2*sx,y2*sy)
            draw.rectangle(box,outline=color,width=3)
            text=prefix+' '+obj['label']+(f' {obj["confidence"]:.2f}' if prefix=='Pred' else '')
            draw.text((box[0],max(26,box[1]-22)),text,fill=color,font=font)
    draw.rectangle((0,0,image.width,25),fill='black')
    draw.text((5,2),'FP32 ONNX | Green: GT | Red: predictions score >= 0.5 | '+item['relative_path'],fill='white',font=font)
    image.save(source/'preview.jpg',quality=90)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=CLASSES,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    weights=ROOT/'weights'/WEIGHTS[args.scene]
    output=(args.output or ROOT/f'outputs/JSTY-YOLO/experiments/{args.scene}_val300/{weights.stem}_val300_20261102').resolve()
    print(json.dumps(dict(scene=args.scene,weights=str(weights),output=str(output),execute=args.execute),ensure_ascii=False),flush=True)
    if not args.execute:
        return
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite prior inference: {output}')
    import cv2
    import onnxruntime as ort
    from industrial_fewshot.detectors.yolo_onnx import YOLOONNXRunner,yolo_onnx_settings,letterbox,decode_heads
    from industrial_fewshot.data.labelme import validate_pixel_prediction
    from industrial_fewshot.evaluation.detection_metrics import check_evaluation_dependencies
    from industrial_fewshot.evaluation.evaluator import evaluate_records
    from industrial_fewshot.evaluation.reporting import format_metrics
    from analyze_box_precision import tolerance_ap
    cv2.setNumThreads(2)
    check_evaluation_dependencies()
    manifest,data,items=cohort(args.scene)
    labels=CLASSES[args.scene]
    settings=yolo_onnx_settings(dict(weights=str(weights),input_size=[960,1280],class_names=labels,
        target_class_id=0,confidence=.001,nms_iou=.7,max_detections=100,intra_op_threads=4,
        providers=['CPUExecutionProvider'],exclude_reference_overlap=False),ROOT)
    runner=YOLOONNXRunner(dict(yolo_onnx=settings,labels=[dict(name=c) for c in labels]))
    assert all(v.type=='tensor(float)' for v in [*runner.session.get_inputs(),*runner.session.get_outputs()])
    meta=runner.session.get_modelmeta().custom_metadata_map
    names=ast.literal_eval(meta['names'])
    assert [names[i] for i in range(len(names))]==labels
    marker=read(ROOT/f'outputs/_shared/datasets/{args.scene}_val300/dataset.json')
    evaluation=dict(iou_threshold=.5,score_threshold=.5,score_thresholds=[.5,.7],max_detections_per_image=100)
    output.mkdir(parents=True)
    report=dict(schema='industrial_fewshot.jsty_fp32_val300.v1',scene_name=f'{args.scene}_val300',
        method='jsty_yolo',backend='yolo_onnx',model=str(weights),run_dir=str(output),image_count=300,
        image_dir=str(manifest.parent/'images'),ground_truth_dir=str(manifest.parent/'images'),
        image_paths=[i['relative_path'] for i in items],dataset_manifest=str(manifest),
        dataset_manifest_sha256=sha(manifest),dataset_sha256=marker['dataset_sha256'],
        missing_ground_truth='error',weights_sha256=sha(weights),onnx_metadata=meta,onnxruntime_version=ort.__version__,
        providers=runner.session.get_providers(),precision='FP32',inference=settings,
        evaluation=evaluation,training_regime='custom_network_supervised',training_shot_per_class=None,
        training_image_count=None,training_test_overlap='not_verified_against_original_training_snapshot',
        inference_shot_per_class=0,experiment=dict(positive_count=0,negative_count=0,variant='监督检测·FP32 ONNX'),
        targets=[dict(id='all_classes',backend='yolo_onnx',model=str(weights),labels=[dict(name=c) for c in labels])],
        decode_policy='One ONNX forward per image. Native class argmax per anchor; DFL16; class-aware NMS 0.7; retain 100/class.',
        status='running',reused_prediction_count=0,executed_inference_count=0,failed_request_count=0)
    write(output/'run.json',report)
    write(output/'config.resolved.json',dict(onnx=settings,evaluation=evaluation,classes=labels))
    (output/'prompt.txt').write_text('No text or visual prompts. Supervised custom-network FP32 ONNX detector.\n')
    started=time.perf_counter();records=[];forward_seconds=0.;decoded_seconds=0.;preview_written=False
    try:
        with (output/'records.jsonl').open('w') as stream:
            for number,item in enumerate(items,1):
                image=cv2.imread(item['path']);assert image is not None,item['path']
                tensor,transform=letterbox(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),settings['input_size'])
                start=time.perf_counter()
                outputs=runner.session.run(None,{runner.input_name:tensor})
                forward_seconds+=time.perf_counter()-start
                start=time.perf_counter();objects=[]
                for class_id,label in enumerate(labels):
                    objects.extend(decode_heads(outputs,dict(settings,target_class_id=class_id),transform,label))
                decoded_seconds+=time.perf_counter()-start
                payload=validate_pixel_prediction(dict(annotation_status='annotated' if objects else 'no_target',
                    reason_code='target_detected' if objects else 'no_target',reason='Custom supervised FP32 ONNX inference',
                    target_count=len(objects),objects=objects),set(labels),item['width'],item['height'])
                record=dict(relative_path=item['relative_path'],image_sha256=item['image_sha256'],
                            target_id='all_classes',status='success',**payload)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();records.append(record)
                if not preview_written and item['ground_truth']:
                    preview(output,item,objects);preview_written=True
                if number%25==0:
                    print(f'{args.scene}: {number}/300 images, elapsed={time.perf_counter()-started:.1f}s',flush=True)
        metrics=evaluate_records(items,records,labels,evaluation)
        geometry=tolerance_ap([{**item,'predictions':r['objects']} for item,r in zip(items,records)],labels)
        metrics['AP_box20']=geometry['AP_box20']
        metrics['class_AP_box20']={k:v['AP_box20'] for k,v in geometry['per_class'].items()}
        for threshold in [.85,.9,.95]:
            metrics[f'AP{round(threshold*100)}']=sum(c['ap_by_iou'][str(threshold)] for c in metrics['per_class'].values())/len(labels)
        assert metrics['evaluation_status']=='complete' and metrics['evaluated_image_count']==300
        write(output/'metrics.json',metrics)
        (output/'evaluation.txt').write_text(format_metrics(metrics)+'\n')
        report.update(status='completed',executed_inference_count=300,success_request_count=300,
            failed_request_count=0,elapsed_seconds=time.perf_counter()-started,
            onnx_forward_seconds=forward_seconds,postprocess_seconds=decoded_seconds)
        print(format_metrics(metrics),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        write(output/'run.json',report)
    print('Saved',output,flush=True)


if __name__=='__main__':
    main()
