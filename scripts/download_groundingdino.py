"""Download an exact GroundingDINO Tiny/Base Hugging Face snapshot to project weights."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['tiny','base'], default='tiny')
    args = p.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download
    repo = f'IDEA-Research/grounding-dino-{args.variant}'
    info = HfApi(token=False).model_info(repo)
    destination = ROOT / 'weights' / f'grounding-dino-{args.variant}'
    snapshot_download(repo, revision=info.sha, local_dir=destination, token=False,
                      allow_patterns=['*.json','*.txt','*.model','*.safetensors'])
    if not (destination/'config.json').is_file() or not any(destination.glob('*.safetensors')):
        raise RuntimeError('Checkpoint incomplete; config.json or safetensors weights missing')
    (destination/'download_manifest.json').write_text(json.dumps(dict(repository=repo, revision=info.sha),indent=2)+'\n',encoding='utf-8')
    print(f'Local checkpoint ready: {destination}')


if __name__ == '__main__':
    main()
