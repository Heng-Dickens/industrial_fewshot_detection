"""Check local runtime imports and native CPU/CUDA operators without model assets."""
import argparse
import os
from pathlib import Path
import sys

from check_model_installation import import_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True,
                        choices=['yoloe', 'groundingdino', 'pet_dino', 'wedetect'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      YOLO_AUTOINSTALL='false',
                      YOLO_CONFIG_DIR=str(root / 'outputs/runtime/yoloe_settings'))
    print(f'Checking {args.backend}: {sys.executable}', flush=True)
    errors = import_checks(args.backend, 'text')
    if errors:
        raise RuntimeError('\n'.join(errors))
    import torch
    from torchvision.ops import nms
    if args.backend == 'yoloe':
        import cv2
        import onnxruntime
        print(f'OpenCV {cv2.__version__}; ONNX Runtime {onnxruntime.__version__}')
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    print(f'CUDA {torch.version.cuda}; device count: {torch.cuda.device_count()}')
    for index in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{index}')
        a = torch.ones((32, 32), device=device)
        assert (a @ a).sum().item() == 32768
        selected = nms(torch.tensor([[0., 0., 10., 10.]], device=device),
                       torch.tensor([0.8], device=device), 0.5)
        assert selected.tolist() == [0]
        if args.backend == 'pet_dino':
            from mmcv.ops import MultiScaleDeformableAttention
            attention = MultiScaleDeformableAttention(
                embed_dims=32, num_heads=4, num_levels=1, num_points=2,
                batch_first=True, dropout=0).to(device).eval()
            with torch.inference_mode():
                result = attention(
                    query=torch.randn(1, 2, 32, device=device),
                    value=torch.randn(1, 4, 32, device=device),
                    reference_points=torch.rand(1, 2, 1, 2, device=device),
                    spatial_shapes=torch.tensor([[2, 2]], device=device),
                    level_start_index=torch.tensor([0], device=device))
            assert result.shape == (1, 2, 32) and torch.isfinite(result).all()
        torch.cuda.synchronize(device)
        print(f'OK cuda:{index} {torch.cuda.get_device_name(index)} matmul/NMS'
              + ('/MMCV attention' if args.backend == 'pet_dino' else ''), flush=True)
    print('Runtime check passed. Model weights, datasets, and API credentials are separate assets.')


if __name__ == '__main__':
    main()
