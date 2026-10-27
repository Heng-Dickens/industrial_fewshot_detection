"""Install isolated Linux Conda runtimes for every supported local backend.

Run with the base Python, not an environment being modified. Weights and data
are prepared separately. Logs and exact installed versions stay under outputs/.
"""
from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'yoloe': 'industrial-fewshot',
    'groundingdino': 'industrial-groundingdino',
    'pet_dino': 'industrial-pet-dino',
    'wedetect': 'industrial-wedetect',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda', default=str(Path.home() / 'miniconda3/bin/conda'))
    parser.add_argument('--backend', choices=[*NAMES, 'all'], default='all')
    parser.add_argument('--status', action='store_true', help='Print recent installation logs without installing')
    parser.add_argument('--verify', action='store_true', help='Check imports and CPU/CUDA operators without weights')
    parser.add_argument('--test-project', action='store_true', help='Run CLI/shared-dependency checks and the existing unittest suite')
    args = parser.parse_args()
    conda = Path(args.conda).resolve()
    envroot = conda.parent.parent / 'envs'
    logs = ROOT / 'outputs/environment_setup'
    logs.mkdir(parents=True, exist_ok=True)
    if args.status:
        for backend, name in NAMES.items():
            print(f'\n{backend}: {envroot / name / "bin/python"}')
            logfile = logs / f'{backend}-install.log'
            print('\n'.join(logfile.read_text(errors='replace').splitlines()[-10:])
                  if logfile.exists() else 'Not started')
        return
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', PIP_PREFER_BINARY='1',
                       YOLO_AUTOINSTALL='false',
                       YOLO_CONFIG_DIR=str(ROOT / 'outputs/runtime/yoloe_settings'))
    selected = NAMES if args.backend == 'all' else [args.backend]
    if args.test_project:
        python = envroot / NAMES['yoloe'] / 'bin/python'
        commands = [[str(python), 'scripts/check_environment.py'],
                    [str(python), 'main.py', '--help'],
                    [str(python), '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.', '-q']]
        with (logs / 'project-tests.log').open('w') as log:
            for command in commands:
                print(shlex.join(command), flush=True)
                log.write('\n$ ' + shlex.join(command) + '\n')
                log.flush()
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    print((logs / 'project-tests.log').read_text(errors='replace')[-10000:])
                    raise SystemExit(result.returncode)
        print('CLI, shared dependencies and existing unittest suite passed.')
        return
    if args.verify:
        for backend in selected:
            python = envroot / NAMES[backend] / 'bin/python'
            with (logs / f'{backend}-runtime-check.log').open('w') as log:
                for command in ([str(python), '-m', 'pip', 'check'],
                                [str(python), str(ROOT / 'scripts/check_conda_runtime.py'),
                                 '--backend', backend]):
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
                    log.flush()
                    if result.returncode:
                        print((logs / f'{backend}-runtime-check.log').read_text(errors='replace'))
                        raise SystemExit(result.returncode)
            print((logs / f'{backend}-runtime-check.log').read_text(errors='replace'), flush=True)
        return
    # Current OpenCV wheels require NumPy 2; PET and the older model code use 1.x.
    common_constraints = ('numpy<2\nopencv-python<4.12\nopencv-python-headless<4.12\n'
                          'setuptools<81\nhuggingface-hub>=0.26.3,<1\n'
                          'pandas==2.2.3\nmatplotlib==3.8.4\n')

    for backend in selected:
        name = NAMES[backend]
        python = envroot / name / 'bin/python'
        torch, vision, cuda = ('2.1.2', '0.16.2', 'cu121') if backend == 'pet_dino' else ('2.5.1', '0.20.1', 'cu124')
        constraints = logs / f'{backend}-constraints.txt'
        constraints.write_text(common_constraints + f'torch=={torch}\ntorchvision=={vision}\n'
                               + (f'torchaudio=={torch}\n' if backend == 'yoloe' else ''), encoding='utf-8')
        with (logs / f'{backend}-install.log').open('a', encoding='utf-8') as log:
            def run(command):
                print(f'[{backend}] {shlex.join(map(str, command))}', flush=True)
                log.write('\n$ ' + shlex.join(map(str, command)) + '\n')
                log.flush()
                subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)

            def pip(*parts):
                run([python, '-m', 'pip', 'install', '-c', constraints, *parts])

            if not python.is_file():
                run([conda, 'create', '-n', name, 'python=3.10', 'pip', '-y'])
            run([python, '-c', 'import sys; assert sys.version_info[:2] == (3,10); print(sys.executable)'])
            pip('pip', 'setuptools<81', 'wheel')
            # Install general packages from PyPI before selecting the PyTorch
            # wheel index, which is not a complete general package repository.
            pip('numpy==1.23.5' if backend == 'pet_dino' else 'numpy==1.26.4',
                'Pillow==11.3.0', 'filelock', 'typing-extensions', 'networkx',
                'jinja2', 'fsspec', 'sympy==1.13.1')
            pip(f'torch=={torch}', f'torchvision=={vision}', '--index-url',
                f'https://download.pytorch.org/whl/{cuda}')
            if backend == 'pet_dino':
                pip('-r', 'requirements-pet-dino.txt')
                pip('mmcv==2.1.0', '--only-binary=mmcv', '--find-links',
                    'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html')
            elif backend == 'wedetect':
                pip('-r', 'requirements-wedetect.txt')
            else:
                pip('-e', '.')
                if backend == 'yoloe':
                    # New PyPI ORT wheels require glibc >=2.27; conda-forge
                    # retains a 1.17.3 build compatible with CentOS 7.
                    libc, version = platform.libc_ver()
                    if libc == 'glibc' and tuple(map(int, version.split('.')[:2])) < (2, 27):
                        run([conda, 'install', '-n', name, 'onnxruntime=1.17.3', 'numpy=1.26.4', '-y'])
                    else:
                        pip('onnxruntime>=1.17,<2')
                    pip('opencv-python<4.12')
                if backend == 'groundingdino':
                    pip('-r', 'requirements-groundingdino.txt')
                else:
                    pip('-e', 'third_party/yoloe')
                    pip('-r', 'requirements-yoloe-text.txt')
                    pip('huggingface-hub>=0.26.3,<1')
            run([python, '-m', 'pip', 'check'])
            with (logs / f'{backend}-freeze.txt').open('w') as freeze:
                subprocess.run([str(python), '-m', 'pip', 'freeze'], env=environment,
                               stdout=freeze, check=True)
        print(f'[{backend}] Installation complete: {python}', flush=True)


if __name__ == '__main__':
    main()
