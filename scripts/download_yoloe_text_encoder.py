"""Download the official MobileCLIP-B(LT) checkpoint used by pinned THU-MIG YOLOE."""
import hashlib
import json
from pathlib import Path
import urllib.request

URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt'
EXPECTED_SIZE = 599214572
EXPECTED_MD5 = 'd61231bd164d23e7e500076e0e8f0d3e'  # Official Apple HTTP ETag.


def checksum(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    destination = Path(__file__).resolve().parents[1] / 'weights/yoloe/mobileclip_blt.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        partial = destination.with_suffix('.pt.part')
        offset = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status != 206:
                offset = 0
            written = offset
            milestone = written // (50 * 1024 * 1024)
            with partial.open('ab' if offset else 'wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    written += len(chunk)
                    if written // (50 * 1024 * 1024) > milestone:
                        milestone = written // (50 * 1024 * 1024)
                        print(f'Downloaded {written / 1024**2:.0f} / {EXPECTED_SIZE / 1024**2:.0f} MiB', flush=True)
        if partial.stat().st_size != EXPECTED_SIZE or checksum(partial, 'md5') != EXPECTED_MD5:
            raise RuntimeError(f'Official checkpoint size/hash mismatch: {partial}')
        partial.replace(destination)
    if destination.stat().st_size != EXPECTED_SIZE or checksum(destination, 'md5') != EXPECTED_MD5:
        raise RuntimeError(f'Existing encoder checkpoint is not the expected official file: {destination}')
    manifest = dict(source=URL, size=EXPECTED_SIZE, md5=EXPECTED_MD5, sha256=checksum(destination, 'sha256'),
                    model='MobileCLIP-B(LT)', upstream='https://github.com/THU-MIG/yoloe')
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Verified encoder: {destination}', flush=True)
    print(f'SHA256: {manifest["sha256"]}', flush=True)


if __name__ == '__main__':
    main()
