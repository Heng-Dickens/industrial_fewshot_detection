"""Dedicated-model setup implementation. No action flags prints a read-only plan."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('yoloe', 'groundingdino', 'pet_dino', 'wedetect')


@dataclass
class Step:
    title: str
    argv: list[str]
    kind: str = 'command'


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=BACKENDS, required=True)
    p.add_argument('--python', default=sys.executable, help='Base Python 3.10/3.11; pip never installs into this base')
    p.add_argument('--mode', choices=['visual', 'text', 'fusion'])
    p.add_argument('--variant', choices=['tiny', 'base', 'large'])
    p.add_argument('--model', choices=[f'yoloe-{f}{s}-seg.pt' for f in ('v8','11') for s in ('s','m','l')])
    p.add_argument('--torch-build', choices=['cpu', 'cu118', 'cu121', 'cu124'], default='cpu')
    p.add_argument('--weights-endpoint', default='https://huggingface.co')
    p.add_argument('--mmcv-wheel', help='PET only: matching local mmcv==2.1.0 wheel')
    p.add_argument('--expected-weights-sha256', help='Optional YOLOE checkpoint hash')
    for flag in ('import-source', 'install-dependencies', 'download-weights', 'check', 'load-model', 'all', 'plan'):
        p.add_argument('--'+flag, action='store_true')
    args = p.parse_args(argv)
    if args.backend == 'yoloe':
        if args.variant:
            p.error('YOLOE uses --model rather than --variant')
        args.mode = args.mode or 'visual'
        args.model = args.model or ('yoloe-v8s-seg.pt' if args.mode == 'visual' else 'yoloe-11l-seg.pt')
    elif args.backend == 'pet_dino':
        if args.variant or args.model or args.mode == 'fusion':
            p.error('PET-DINO uses Swin-T, --mode visual/text; no --variant/--model/fusion')
        if args.torch_build == 'cu124':
            p.error('PET-DINO torch 2.1.2 supports cpu/cu118/cu121, not cu124')
        args.mode = args.mode or 'visual'
    else:
        if args.mode not in (None, 'text') or args.model:
            p.error('This backend only supports text mode; --model is for YOLOE')
        args.mode = 'text'
        args.variant = args.variant or ('tiny' if args.backend == 'groundingdino' else 'base')
        if args.backend == 'groundingdino' and args.variant == 'large':
            p.error('GroundingDINO supports tiny/base')
    if args.mmcv_wheel and args.backend != 'pet_dino':
        p.error('--mmcv-wheel is only for PET-DINO')
    if args.mmcv_wheel and (not Path(args.mmcv_wheel).name.startswith('mmcv-2.1.0-') or not args.mmcv_wheel.endswith('.whl')):
        p.error('--mmcv-wheel must name a compiled mmcv-2.1.0-*.whl matching this Python/platform/CUDA')
    if args.expected_weights_sha256 and (args.backend != 'yoloe' or len(args.expected_weights_sha256) != 64
            or any(c not in '0123456789abcdef' for c in args.expected_weights_sha256.lower())):
        p.error('--expected-weights-sha256 requires YOLOE and 64 hexadecimal characters')
    explicit = any((args.import_source, args.install_dependencies, args.download_weights, args.check, args.load_model))
    if not explicit and not args.all:
        args.plan, args.all = True, True
    if args.all:
        args.import_source = args.install_dependencies = args.download_weights = args.check = True
    if args.load_model:
        args.check = True
    return args


def environment_python(root, backend):
    return Path(root) / f'.venv-{backend}' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def build_plan(args, root=ROOT):
    root = Path(root).resolve()
    target = environment_python(root, args.backend)
    python, scripts, steps = str(target), root/'scripts', []
    def command(title, *parts, kind='command'):
        steps.append(Step(title, [str(part) for part in parts], kind))
    def pip(title, *parts):
        command(title, python, '-m', 'pip', 'install', *parts)
    if args.backend != 'groundingdino' and (args.import_source or args.install_dependencies or args.download_weights):
        command('Import/verify pinned upstream source', args.python, scripts/'import_prompt_detectors.py', '--backend', args.backend)
    if args.install_dependencies or args.download_weights:
        command('Create environment if absent (no system-site-packages)', args.python, '-m', 'venv', target.parent.parent, kind='environment')
    if args.install_dependencies:
        pip('Prepare packaging tools', 'pip', 'setuptools', 'wheel')
        torch, vision = ('2.1.2', '0.16.2') if args.backend == 'pet_dino' else ('2.5.1', '0.20.1')
        command('Prepare PyTorch; preserve working YOLOE/GroundingDINO builds', python, '-m', 'pip', 'install',
                f'torch=={torch}', f'torchvision=={vision}', '--index-url', f'https://download.pytorch.org/whl/{args.torch_build}', kind='torch')
        if args.backend == 'pet_dino':
            pip('Install PET-DINO dependencies', '-r', root/'requirements-pet-dino.txt')
            wheel = [str(Path(args.mmcv_wheel).expanduser().resolve())] if args.mmcv_wheel else [
                'mmcv==2.1.0', '--find-links', f'https://download.openmmlab.com/mmcv/dist/{args.torch_build}/torch2.1/index.html']
            command('Install compiled MMCV wheel (never mmcv-lite or implicit source builds)', python, '-m', 'pip', 'install',
                    '--only-binary=mmcv', '--constraint', root/'requirements-pet-dino.txt', *wheel, kind='mmcv')
        elif args.backend == 'wedetect':
            pip('Install standalone WeDetect dependencies (no mmdet/mmcv)', '-r', root/'requirements-wedetect.txt')
        else:
            pip('Install shared experiment dependencies', '-r', root/'requirements.txt')
            if args.backend == 'groundingdino':
                pip('Install GroundingDINO Transformers API', '-r', root/'requirements-groundingdino.txt')
            else:
                pip('Install pinned YOLOE source', '-e', root/'third_party/yoloe')
                pip('Install download client', 'huggingface-hub>=0.26.3,<1')
                if args.mode != 'visual':
                    pip('Install MobileCLIP text/fusion dependencies', '-r', root/'requirements-yoloe-text.txt')
    elif args.download_weights:
        pip('Install lightweight download client in target environment', 'huggingface-hub>=0.26.3,<1')
    if args.download_weights:
        if args.backend == 'groundingdino':
            command('Download/verify GroundingDINO checkpoint', python, scripts/'download_groundingdino.py', '--variant', args.variant)
        elif args.backend == 'yoloe':
            download = [python, str(scripts/'download_yoloe_weights.py'), '--model', args.model]
            if args.expected_weights_sha256:
                download += ['--expected-sha256', args.expected_weights_sha256]
            command('Download/verify YOLOE detection weights', *download)
            if args.mode != 'visual':
                command('Download/verify MobileCLIP text weights', python, scripts/'download_yoloe_text_encoder.py')
        else:
            command('Download detector and language assets', python, scripts/'download_prompt_detector_weights.py', '--backend', args.backend,
                    '--variant', args.variant or 'base')
    if args.check:
        check = [python, str(scripts/'check_model_installation.py'), '--backend', args.backend, '--mode', args.mode,
                 '--variant', args.variant or 'base']
        if args.model:
            check += ['--model', args.model]
        if args.load_model:
            check += ['--load-model']
        command('Check source, imports, native operators and assets offline', *check)
    return steps


def run_step(step, args, root=ROOT):
    target = environment_python(root, args.backend)
    if step.kind == 'environment':
        if target.is_file():
            envroot = target.parent.parent.resolve()
            if not (envroot/'pyvenv.cfg').is_file():
                raise RuntimeError(f'Existing target is not a verified venv (pyvenv.cfg missing): {envroot}')
            probe = subprocess.run([str(target), '-c', 'import sys,json;print(json.dumps(sys.prefix))'],
                                   capture_output=True,text=True,timeout=20,check=True,cwd=root)
            if Path(json.loads(probe.stdout)).resolve() != envroot:
                raise RuntimeError(f'Target Python resolves to another environment; refusing package changes: {target}')
            print(f'Keeping existing environment: {target}', flush=True)
            return
        if target.parent.parent.exists():
            raise RuntimeError(f'Incomplete environment exists: {target.parent.parent}; inspect before retrying')
    if step.kind == 'torch' and args.backend in {'yoloe','groundingdino'}:
        probe = subprocess.run([str(target), '-c', 'import torch,torchvision;print(torch.__version__)'],
                               capture_output=True, text=True, timeout=60, cwd=root)
        if probe.returncode == 0:
            print(f'Keeping installed PyTorch {probe.stdout.strip()}; --torch-build applies to fresh installs.', flush=True)
            return
    env = {**os.environ, 'PYTHONIOENCODING':'utf-8', 'HF_ENDPOINT':args.weights_endpoint,
           'YOLO_AUTOINSTALL':'false', 'YOLO_CONFIG_DIR':str(root/'outputs/runtime/yoloe_settings')}
    result = subprocess.run(step.argv, cwd=root, env=env)
    if result.returncode:
        if step.kind == 'mmcv':
            raise RuntimeError('Compiled MMCV installation failed. Match Python 3.10/3.11, torch 2.1.2 and cpu/cu118/cu121; '
                               'use --mmcv-wheel PATH for a matching local wheel. mmcv-lite is not a substitute. '
                               'See docs/guides/MODEL_INSTALLATION.md.')
        raise RuntimeError(f'{step.title} failed (exit {result.returncode}); later steps were not executed')


def main(argv=None):
    args = options(argv)
    steps = build_plan(args)
    print(f'Model: {args.backend}; mode: {args.mode}; environment: {environment_python(ROOT,args.backend)}', flush=True)
    if args.plan:
        print('PLAN ONLY: no commands will execute.', flush=True)
        for step in steps:
            print(step.title + '\n  ' + subprocess.list2cmdline(step.argv))
        return 0
    if args.install_dependencies or args.download_weights or args.import_source:
        version = subprocess.run([args.python, '-c', 'import sys,json;print(json.dumps(list(sys.version_info[:2])))'],
                                 capture_output=True, text=True, timeout=20, check=True)
        if json.loads(version.stdout) not in ([3,10],[3,11]):
            raise RuntimeError('Use a Python 3.10/3.11 base interpreter via -Python/--python')
    for step in steps:
        print('\n'+step.title, flush=True)
        run_step(step,args)
    print('Requested setup steps completed. Imports/assets checks do not measure inference quality.', flush=True)
    if args.backend in {'pet_dino','wedetect'}:
        print('Run main.py using the project environment; YAML worker_python selects the isolated model environment.')
    else:
        print(f'Inference Python: {environment_python(ROOT,args.backend)}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
