"""Industrial few-shot detection experiments; old --config commands remain valid."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from industrial_fewshot.cli import main


def project_runtime(argv):
    """Recover a normal inference launch when `python` resolves to an incomplete base."""
    if ('--config' not in argv or any(flag in argv for flag in ('--dry-run', '--evaluate-only', '--help', '-h'))
            or os.environ.get('INDUSTRIAL_USE_CURRENT_PYTHON') == '1'):
        return None
    root = Path(__file__).resolve().parent
    candidate = root / '.venv-yoloe/Scripts/python.exe'
    if not candidate.is_file() or Path(sys.executable).resolve() == candidate.resolve():
        return None
    names = ('PIL', 'numpy', 'scipy', 'yaml', 'pycocotools', 'openai')
    if all(importlib.util.find_spec(name) is not None for name in names):
        return None
    check = subprocess.run([str(candidate), str(root / 'scripts/check_environment.py')],
                           capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=30)
    return candidate if check.returncode == 0 else None


if __name__ == "__main__":
    runtime = project_runtime(sys.argv[1:])
    if runtime is not None:
        print(f'当前 Python 依赖不完整：{sys.executable}；改用已验证的项目环境：{runtime}', file=sys.stderr, flush=True)
        from industrial_fewshot.runtime import run_project_child
        raise SystemExit(run_project_child(runtime, Path(__file__).resolve(), sys.argv[1:]))
    raise SystemExit(main())
