"""Check the selected Python's shared experiment dependencies without inference."""
import importlib
import importlib.metadata
from pathlib import Path
import sys


def main():
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        print(__doc__)
        return 0
    print(f'Python: {sys.executable}', flush=True)
    requirements = [('PIL', 'Pillow'), ('numpy', 'numpy'), ('scipy.optimize', 'scipy'),
                    ('yaml', 'PyYAML'), ('pycocotools.cocoeval', 'pycocotools'), ('openai', 'openai')]
    failed = False
    for module, distribution in requirements:
        try:
            imported = importlib.import_module(module)
            if module == 'openai':
                getattr(imported, 'AsyncOpenAI')
            print(f'OK {distribution} {importlib.metadata.version(distribution)}', flush=True)
        except Exception as exc:
            failed = True
            print(f'FAIL {distribution}: {type(exc).__name__}: {exc}', flush=True)
    if failed:
        requirements_path = Path(__file__).resolve().parents[1] / 'requirements.txt'
        print(f'Install using this interpreter: "{sys.executable}" -m pip install -r "{requirements_path}"')
    else:
        print('All shared dependencies import successfully. No model requests were made.')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
