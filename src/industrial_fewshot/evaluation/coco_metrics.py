"""Official COCO bbox evaluation, using in-memory images, GT and predictions."""
import contextlib
import io


STAT_NAMES = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
              'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')


def f2_score(precision: float, recall: float) -> float:
    denominator = 4 * precision + recall
    return 5 * precision * recall / denominator if denominator else 0.0


def evaluate_coco(samples: list[dict], labels: list[str]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{'id': index + 1, 'name': label} for index, label in enumerate(labels)]
    category_ids = {row['name']: row['id'] for row in categories}
    images, annotations, detections = [], [], []
    for image_id, sample in enumerate(samples, 1):
        images.append({'id': image_id, 'width': sample['width'], 'height': sample['height']})
        for source, destination in (('ground_truth', annotations), ('predictions', detections)):
            for obj in sample[source]:
                if obj['label'] not in category_ids:
                    continue
                x1, y1, x2, y2 = obj['bbox_pixels_xyxy']
                box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                row = {'id': len(destination) + 1, 'image_id': image_id,
                       'category_id': category_ids[obj['label']], 'bbox': box,
                       'area': box[2] * box[3], 'iscrowd': 0}
                if source == 'predictions':
                    row['score'] = float(obj['confidence'])
                destination.append(row)

    # Build the empty-result case explicitly: COCO.loadRes([]) indexes anns[0].
    # Suppress setup/progress chatter, retaining only the standard 12-line summary.
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth, predictions = COCO(), COCO()
        for dataset, rows in ((ground_truth, annotations), (predictions, detections)):
            dataset.dataset = {'images': images, 'categories': categories, 'annotations': rows}
            dataset.createIndex()
        evaluator = COCOeval(ground_truth, predictions, 'bbox')
        evaluator.params.imgIds = [image['id'] for image in images]
        evaluator.params.catIds = [row['id'] for row in categories]
        evaluator.evaluate()
        evaluator.accumulate()
        summary = io.StringIO()
        with contextlib.redirect_stdout(summary):
            evaluator.summarize()

    per_class = {}
    for index, label in enumerate(labels):
        precision_values = evaluator.eval['precision'][0, :, index, 0, 2]
        defined = precision_values[precision_values > -1]
        tp = dets = gts = 0
        for image in evaluator.evalImgs:
            if image is None or image['category_id'] != category_ids[label] or image['aRng'] != evaluator.params.areaRng[0]:
                continue
            valid = ~image['dtIgnore'][0]
            tp += int(((image['dtMatches'][0] > 0) & valid).sum())
            dets += int(valid.sum())
            gts += int((~image['gtIgnore'].astype(bool)).sum())
        precision, recall = tp / dets if dets else 0.0, tp / gts if gts else 0.0
        per_class[label] = {'AP50': float(defined.mean()) if defined.size else None,
                            'recall': recall if gts else None, 'precision': precision,
                            'f2': f2_score(precision, recall) if gts else None,
                            'tp': tp, 'fp': dets - tp, 'fn': gts - tp}
    return {'engine': 'pycocotools.COCOeval', 'iou_type': 'bbox',
            'stats': dict(zip(STAT_NAMES, (float(value) for value in evaluator.stats))),
            'summary': summary.getvalue().rstrip(), 'per_class': per_class,
            'policy': 'Official COCO bbox evaluation; area is bbox width*height in original pixels; '
                      'IoU=0.50:0.95, maxDets=1/10/100, all prediction scores. '
                      'LabelMe input has no crowd/ignore flags. Category AP/Recall are at IoU=0.50, '
                      'maxDets=100; F2 uses final TP/FP precision and recall, not AP as precision.'}
