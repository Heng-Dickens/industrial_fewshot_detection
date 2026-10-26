"""Repair project-directory references in the existing Windows YOLOE environment.

Run with .venv-yoloe/Scripts/python.exe; no downloads or dependency changes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    environment = root / '.venv-yoloe'
    if Path(sys.prefix).resolve() != environment or sys.platform != 'win32':
        parser.error('Run with this project\'s .venv-yoloe/Scripts/python.exe on Windows.')
    old = root.with_name('annotationtool')
    replacements = [(str(old), str(root)), (old.as_posix(), root.as_posix()),
                    (json.dumps(str(old))[1:-1], json.dumps(str(root))[1:-1])]
    packages = environment / 'Lib/site-packages'
    scripts = environment / 'Scripts'
    candidates = [scripts / 'activate', scripts / 'activate.bat',
                  *packages.glob('*.pth'), *packages.glob('__editable__*finder.py'),
                  *packages.glob('*.dist-info/direct_url.json')]
    updated = []
    for path in candidates:
        if not path.is_file():
            continue
        original = path.read_bytes()
        revised = original
        for source, target in replacements:
            revised = revised.replace(source.encode('utf-8'), target.encode('utf-8'))
        if revised != original:
            if not args.dry_run:
                path.write_bytes(revised)
            updated.append(str(path.relative_to(root)))

    # Recreate distlib launchers from installed entry-point metadata rather than
    # editing executable bytes. pip3 and pip3.X are generated aliases of pip.
    from pip._vendor.distlib.scripts import ScriptMaker
    maker = ScriptMaker(None, str(scripts))
    maker.executable, maker.clobber, maker.variants = sys.executable, True, {''}
    for distribution in importlib.metadata.distributions(path=[str(packages)]):
        for entry in distribution.entry_points:
            if entry.group not in {'console_scripts', 'gui_scripts'}:
                continue
            names = [entry.name]
            if entry.name == 'pip':
                names += ['pip3', f'pip{sys.version_info.major}.{sys.version_info.minor}']
            for name in names:
                launcher = scripts / (name + '.exe')
                if launcher.is_file() and any(source.encode('utf-8') in launcher.read_bytes()
                                               for source, _ in replacements):
                    if not args.dry_run:
                        maker.make(f'{name} = {entry.value}', {'gui': entry.group == 'gui_scripts'})
                    updated.append(str(launcher.relative_to(root)))
    print(json.dumps({'dry_run': args.dry_run, 'updated': updated}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
