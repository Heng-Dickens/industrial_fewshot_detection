"""Compatibility imports; implementations live in src/industrial_fewshot."""
import importlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_modules = {'pipeline': 'industrial_fewshot.runner', 'scene_config': 'industrial_fewshot.config', 'prediction_cache': 'industrial_fewshot.cache', 'experiments': 'industrial_fewshot.experiments', 'labelme': 'industrial_fewshot.data.labelme', 'detection_metrics': 'industrial_fewshot.evaluation.detection_metrics', 'evaluation': 'industrial_fewshot.evaluation.evaluator', 'inference': 'industrial_fewshot.detectors.mllm', 'yoloe_backend': 'industrial_fewshot.detectors.yoloe_visual', 'reference_suite': 'industrial_fewshot.data.supports', 'references': 'industrial_fewshot.prompts.references'}
for _name, _target in _modules.items():
    _module = importlib.import_module(_target)
    sys.modules[__name__ + "." + _name] = _module
    globals()[_name] = _module
