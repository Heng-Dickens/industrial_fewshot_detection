"""Import pinned official detector sources without executing upstream setup scripts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOS = {'pet_dino': ('fuweifuvtoo/PET_DINO', '7830a462f9a320f34293cce0aabdda1256d9dc15'),
         'wedetect': ('WeChatCV/WeDetect', '42c34c6742a22f26280bb0e59f95a701baee1a32'),
         'yoloe': ('THU-MIG/yoloe', '40cd606cabdbe2b566d6f14a6b162c89206e9a1b')}
ENTRIES = {'pet_dino':'mmdet/apis/det_inferencer.py', 'wedetect':'deploy/test_coco_pytorch.py',
           'yoloe':'ultralytics/models/yolo/yoloe/predict_vp.py'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=list(REPOS) + ['all'], help='Omit to preserve legacy PET-DINO + WeDetect import')
    args = parser.parse_args(argv)
    selected = list(REPOS) if args.backend == 'all' else ['pet_dino','wedetect'] if args.backend is None else [args.backend]
    for name in selected:
        repo, revision = REPOS[name]
        dest = ROOT / 'third_party' / name
        if dest.exists():
            if not (dest / 'UPSTREAM.json').is_file():
                raise RuntimeError(f'{dest} exists without provenance; inspect the incomplete import first')
            if not (dest / ENTRIES[name]).is_file():
                raise RuntimeError(f'{dest} is incomplete; upstream entry point is missing')
            metadata = json.loads((dest/'UPSTREAM.json').read_text(encoding='utf-8'))
            if metadata.get('revision') != revision:
                raise RuntimeError(f'{dest} is a different revision; refusing to overwrite existing source')
            print(f'{name}: already exists; leaving source unchanged')
            continue
        data = urllib.request.urlopen(f'https://codeload.github.com/{repo}/zip/{revision}', timeout=180).read()
        archive = zipfile.ZipFile(io.BytesIO(data))
        prefix = archive.namelist()[0].split('/')[0] + '/'
        # Validate every destination before creating anything.
        members = [(info, (dest / info.filename[len(prefix):]).resolve())
                   for info in archive.infolist() if info.filename != prefix]
        if any(not path.is_relative_to(dest.resolve()) for _, path in members):
            raise ValueError('Unsafe archive member')
        for info, path in members:
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(info))
        metadata = {'repository': f'https://github.com/{repo}', 'revision': revision,
                    'archive_sha256': hashlib.sha256(data).hexdigest(), 'local_modifications': []}
        (dest / 'UPSTREAM.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        print(f'{name}: {revision}, {len(members)} files/directories')


if __name__ == '__main__':
    main()
