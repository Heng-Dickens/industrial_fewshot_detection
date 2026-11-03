"""Canonical locations for comparison reports; runtime IDs stay unchanged."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Logical experiment IDs are kept for existing --stamp and runtime/checkpoint records.
REPORT_LOCATIONS = {
    'yoloe_chache_frozen_savpe_20261030': 'YOLOE/finetune/chache_val300/comparisons/yoloe_search_20261029',
    'yoloe_diaowu_frozen_savpe_20261030': 'YOLOE/finetune/diaowu_val300/comparisons/yoloe_search_20261029',
    'aligned_baselines_20261027': '_shared/comparisons/aligned_baselines_20261027',
    'pet_dino_shots_20261027': 'PET-DINO/comparisons/pet_dino_shot_sweep_20261027',
    'wedetect_chache_prompts_20261027': 'WeDetect/comparisons/chache_val300/wedetect_prompt_sweep_20261027',
    'iou85_box20_20261028': '_shared/comparisons/iou85_box20_20261028',
    'yoloe_finetune_chache_20261029': 'YOLOE/finetune/chache_val300/comparisons/yoloe_search_20261029',
    'yoloe_search_20261029_r2': 'YOLOE/finetune/chache_val300/comparisons/yoloe_search_20261029',
    'yoloe_diaowu_search_20261029': 'YOLOE/finetune/diaowu_val300/comparisons/yoloe_search_20261029',
    'yoloe_search_20261029': '_shared/archive/comparisons/finetuning/chache_val300/yoloe_search_20261029_attempt01_failed',
}


def comparison_dir(experiment_id, *, project_root=PROJECT_ROOT, category='fewshot', dataset='cross_dataset'):
    """Resolve legacy IDs without re-creating flat report directories."""
    if not experiment_id or Path(experiment_id).name != experiment_id or experiment_id in {'.', '..'}:
        raise ValueError('Experiment ID must be a simple directory name.')
    relative = REPORT_LOCATIONS.get(experiment_id)
    if relative is None:
        if category not in {'fewshot','finetuning','localization','archive'}:
            raise ValueError('Unknown comparison category.')
        if Path(dataset).name != dataset or dataset in {'', '.', '..'}:
            raise ValueError('Dataset must be a simple directory name.')
        relative = (f'YOLOE/finetune/{dataset}/comparisons/{experiment_id}' if category == 'finetuning'
                    else f'_shared/comparisons/{experiment_id}')
    return Path(project_root)/'outputs'/relative


def optimization_report_dir(stamp, *, project_root=PROJECT_ROOT, scene=None):
    if scene is None:
        state = Path(project_root)/'outputs/_shared/runtime'/stamp/'state.json'
        if state.is_file():
            scene = json.loads(state.read_text(encoding='utf-8-sig')).get('scene')
        scene = scene or ('diaowu' if stamp.startswith('yoloe_diaowu_') else 'chache')
    if scene not in {'chache','diaowu'}:
        raise ValueError('Unknown fine-tuning scene.')
    return comparison_dir(stamp,project_root=project_root,category='finetuning',dataset=f'{scene}_val300')
