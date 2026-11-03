"""Small-data visual-prompt adapter for the pinned THU-MIG YOLOE source.

Imported by train.py --method yoloe-vp --execute. Supports same-image and
cross-image training, with configurable trainable scope and detection loss.
Dev queries use prototypes extracted ONLY from fixed training-support images.
"""
from copy import copy
import random

import torch
from torch.nn import functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel


def pad_visuals(visuals, class_count):
    """Keep fixed global class IDs even if a batch lacks the last class."""
    if visuals.ndim != 4 or visuals.shape[1] > class_count:
        raise ValueError(f'Unexpected visual prompt tensor: {visuals.shape}')
    return F.pad(visuals, (0, 0, 0, 0, 0, class_count - visuals.shape[1]))


class SupportVPValidator(YOLOEDetectValidator):
    @torch.inference_mode()
    def get_visual_pe(self, model):
        """Override the upstream hard-coded LVIS reference set with this run's train set."""
        data = check_det_dataset(self.args.data)
        count = len(data['names'])
        if count < 1:
            raise ValueError('Visual prompting requires at least one configured class.')
        dataset = build_yolo_dataset(self.args, data['train'], 1, data, mode='val',
            stride=max(int(model.stride.max()), 32), rect=False, load_vp=True)
        loader = build_dataloader(dataset, 1, self.args.workers, shuffle=False, rank=-1)
        sums = torch.zeros(count, model.model[-1].embed, device=self.device)
        occurrences = torch.zeros(count, device=self.device)
        for batch in loader:
            # This batch contains TRAINING supports. Dev-query boxes are never passed here.
            batch = self.preprocess(batch)
            classes = batch['cls'].reshape(-1).long().unique(sorted=True)
            if not classes.numel():
                continue
            visuals = pad_visuals(batch['visuals'].to(self.device), count)
            embedding = model.get_visual_pe(batch['img'], visual=visuals)
            if embedding.shape[:2] != (1, count):
                raise ValueError(f'Unexpected visual embeddings: {embedding.shape}')
            vectors = embedding[0, classes]
            if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) < 1e-8).any():
                raise ValueError('An annotated support produced an invalid visual embedding.')
            sums[classes] += F.normalize(vectors, dim=-1)
            occurrences[classes] += 1
        if (occurrences == 0).any():
            raise ValueError('Training supports must cover every configured dev-query class.')
        means = sums / occurrences[:, None]
        if (means.norm(dim=-1) < 1e-8).any():
            raise ValueError('Aggregated support embedding is degenerate.')
        # Normalize each support, mean, then normalize again, matching the baseline policy.
        return F.normalize(means, dim=-1).unsqueeze(0)


class CrossImageYOLOEModel(YOLOEModel):
    """Support-only prototypes with different training images for support and query.

    Checkpoints require this module on sys.path (provided by the example entrypoints).
    Prediction remains the upstream YOLOE prediction implementation.
    """
    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = v8DetectionLoss(self)
        if preds is None and 'support_img' in batch:
            # Only the parent flag gates return_vpe; preserve each module's BN mode.
            previous = self.training
            self.training = False
            try:
                embeddings = self.predict(batch['support_img'], vpe=batch['support_visuals'], return_vpe=True)
            finally:
                self.training = previous
            classes = batch['support_classes']
            prototype = embeddings[torch.arange(len(classes), device=classes.device), classes]
            prototype = F.normalize(prototype, dim=-1).unsqueeze(0)
            preds = self.predict(batch['img'], vpe=prototype)
        return super().loss(batch, preds)


class SupportVPTrainer(DetectionTrainer):
    """Retain the unfused visual head; train.py supplies the requested freeze list.

    Ordinary YOLODataset keeps the configured class IDs. It avoids the pinned
    YOLOMultiModalDataset's RandomLoadText and its large pretraining cache files.
    Loss/NMS/model remain the upstream detection implementations.
    """
    def __init__(self, *args, adaptation=None, **kwargs):
        self.adaptation = adaptation or {}
        self.support_rng = random.Random()
        self.support_dataset = None
        super().__init__(*args, **kwargs)
        self.support_rng.seed(self.args.seed)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model_type = CrossImageYOLOEModel if self.adaptation.get('protocol') == 'cross_image' else YOLOEModel
        model = model_type(cfg['yaml_file'] if isinstance(cfg, dict) else cfg,
                           ch=3, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode='train', batch=None):
        stride = max(int(de_parallel(self.model).stride.max()) if self.model else 0, 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode,
                                  rect=False, stride=stride, multi_modal=False,
                                  load_vp=mode == 'train')

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        batch['visuals'] = pad_visuals(batch['visuals'].to(self.device), self.data['nc'])
        if self.adaptation.get('protocol') == 'cross_image':
            if self.support_dataset is None:
                self.support_dataset = build_yolo_dataset(self.args, self.data['train'], 1, self.data,
                    mode='val', rect=False, stride=32, multi_modal=False, load_vp=True)
                self.support_indices = {i: [] for i in range(self.data['nc'])}
                for index, label in enumerate(self.support_dataset.labels):
                    for cls in set(label['cls'].reshape(-1).astype(int).tolist()):
                        self.support_indices[cls].append(index)
            query_paths = set(batch['im_file'])
            supports = []
            for cls in range(self.data['nc']):
                choices = [i for i in self.support_indices[cls]
                           if self.support_dataset.im_files[i] not in query_paths]
                if not choices:
                    raise ValueError(f'No cross-image support for class {cls}; reduce batch size.')
                supports.append(self.support_dataset[self.support_rng.choice(choices)])
            batch['support_img'] = torch.stack([s['img'] for s in supports]).to(self.device).float() / 255.
            batch['support_visuals'] = torch.stack([
                pad_visuals(s['visuals'].unsqueeze(0), self.data['nc'])[0] for s in supports]).to(self.device)
            batch['support_classes'] = torch.arange(self.data['nc'], device=self.device)
        return batch

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # The upstream TVPDetectLoss expects concatenated text+visual channels.
        # With five visual channels only, it would compute vnc=0 and return zero.
        # Ordinary v8DetectionLoss fits our five fixed global IDs; load_vp=True
        # keeps the upstream policy of optimizing its classification component.
        self.model.criterion = v8DetectionLoss(self.model)
        if self.adaptation.get('loss') == 'detection':
            self.model.criterion.hyp = copy(self.model.args)
            self.model.criterion.hyp.load_vp = False
        if self.ema is not None:
            self.ema.ema.criterion = v8DetectionLoss(self.ema.ema)
            if self.adaptation.get('loss') == 'detection':
                self.ema.ema.criterion.hyp = copy(self.ema.ema.args)
                self.ema.ema.criterion.hyp.load_vp = False

    def get_validator(self):
        self.loss_names = 'box', 'cls', 'dfl'
        return SupportVPValidator(self.test_loader, save_dir=self.save_dir,
                                  args=copy(self.args), _callbacks=self.callbacks)
