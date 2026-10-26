"""Lazy model registry; planning/evaluation never imports heavyweight SDKs."""
LOCAL_BACKENDS = {"yoloe", "yoloe_text", "yoloe_fusion", "groundingdino", "yolo_onnx", "pet_dino", "wedetect"}


def is_local(backend: str) -> bool:
    return backend in LOCAL_BACKENDS


def local_runner_class(backend: str):
    if backend in {'pet_dino', 'wedetect'}:
        from .prompt_native import PromptNativeRunner
        return PromptNativeRunner
    if backend == 'yoloe_fusion':
        from .yoloe_fusion import YOLOEFusionRunner
        return YOLOEFusionRunner
    if backend == 'yoloe_text':
        from .yoloe_text import YOLOETextRunner
        return YOLOETextRunner
    if backend == 'yolo_onnx':
        from .yolo_onnx import YOLOONNXRunner
        return YOLOONNXRunner
    if backend == "yoloe":
        from .yoloe_visual import YOLOERunner
        return YOLOERunner
    if backend == "groundingdino":
        from .groundingdino_text import GroundingDINORunner
        return GroundingDINORunner
    raise ValueError(f"不是本地检测器: {backend}")


def create_detector(client, target):
    if target['backend'] in {'yolo_onnx', 'yoloe_text', 'yoloe_fusion', 'pet_dino', 'wedetect'}:
        from .base import LocalDetector
        return LocalDetector(client, target)
    if target["backend"] == "yoloe":
        from .yoloe_visual import YOLOEVisualDetector
        return YOLOEVisualDetector(client, target)
    if target["backend"] == "groundingdino":
        from .groundingdino_text import GroundingDINOTextDetector
        return GroundingDINOTextDetector(client, target)
    from .mllm import MLLMDetector
    return MLLMDetector(client, target)
