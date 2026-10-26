"""Download an official THU-MIG YOLOE checkpoint and verify its upstream SHA256."""
import argparse
import hashlib
import json
from pathlib import Path


def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='yoloe-v8l-seg.pt', choices=[
        f'yoloe-{family}{size}-seg.pt' for family in ('v8', '11') for size in ('s', 'm', 'l')])
    parser.add_argument('--expected-sha256', help='Optional additional expected checkpoint SHA256')
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'jameslahm/yoloe'
    info = HfApi(token=False).model_info(repo, files_metadata=True)
    entry = next(row for row in info.siblings if row.rfilename == args.model)
    expected = entry.lfs.sha256
    if args.expected_sha256 and args.expected_sha256.lower() != expected.lower():
        raise RuntimeError('Explicit checkpoint SHA256 differs from the official release; no download performed')
    destination = Path(__file__).resolve().parents[1] / 'weights/yoloe'
    destination.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {args.model}: {entry.size} bytes, revision {info.sha}', flush=True)
    path = Path(hf_hub_download(repo, args.model, revision=info.sha, local_dir=str(destination), token=False))
    actual = checksum(path)
    if actual != expected or path.stat().st_size != entry.size:
        raise RuntimeError(f'Checkpoint size/SHA256 differs from upstream: {path}')
    manifest = dict(repository=repo, revision=info.sha, filename=args.model, size=entry.size, sha256=actual,
                    source=f'https://huggingface.co/{repo}/resolve/{info.sha}/{args.model}',
                    implementation='https://github.com/THU-MIG/yoloe')
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Verified checkpoint: {path}', flush=True)
    print(f'SHA256: {actual}', flush=True)


if __name__ == '__main__':
    main()
