"""Locations shared by inference, fine-tuning and saved-result reporting."""
from pathlib import Path
from .paths import PROJECT_ROOT

METHODS = {'yoloe': 'YOLOE', 'yoloe_text': 'YOLOE', 'yoloe_fusion': 'YOLOE',
           'pet_dino': 'PET-DINO', 'wedetect': 'WeDetect', 'groundingdino': 'GroundingDINO',
           'yolo_onnx': 'JSTY-YOLO', 'OVD-MLLM': 'OVD-MLLM'}


def method_name(backend):
    return METHODS.get(backend, 'Qwen')


def experiments_dir(method, dataset, *, project_root=PROJECT_ROOT):
    return Path(project_root) / 'outputs' / method / 'experiments' / dataset


def dataset_index_dir(dataset, *, project_root=PROJECT_ROOT):
    return Path(project_root) / 'outputs/_shared/datasets' / dataset


def finetune_dir(scene, *, project_root=PROJECT_ROOT):
    return Path(project_root) / 'outputs/YOLOE/finetune' / f'{scene}_val300' / 'runs'
