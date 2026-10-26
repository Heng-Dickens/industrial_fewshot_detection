"""Download official checkpoints and local tokenizer assets for the new adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['pet_dino', 'wedetect'], required=True)
    parser.add_argument('--variant', choices=['tiny', 'base', 'large'], default='base', help='WeDetect size')
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    target = ROOT / 'weights' / args.backend
    target.mkdir(parents=True, exist_ok=True)
    if args.backend == 'pet_dino':
        repo, filename, name = 'fuweifu/PET-DINO', 'pet_dino_swin-t_8xb4_12e_obj365.pth', 'pet_dino_swin_t.pth'
    else:
        repo = 'fushh7/WeDetect'
        filename = name = f'wedetect_{args.variant}.pth'
    checkpoint = target / name
    info = HfApi(token=False).model_info(repo,files_metadata=True)
    entry = next(row for row in info.siblings if row.rfilename == filename)
    expected = entry.lfs.sha256
    if checkpoint.exists():
        print(f'Keeping existing checkpoint: {checkpoint}')
    else:
        cached = hf_hub_download(repo, filename,revision=info.sha,token=False)
        temporary = checkpoint.with_suffix('.part')
        shutil.copyfile(cached, temporary)
        temporary.replace(checkpoint)
        print(checkpoint)
    checksum = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            checksum.update(chunk)
    if checkpoint.stat().st_size != entry.size or checksum.hexdigest() != expected:
        raise RuntimeError(f'Existing/downloaded checkpoint differs from official size/SHA256; not overwritten: {checkpoint}')
    checkpoint.with_suffix('.manifest.json').write_text(json.dumps(dict(repository=repo,revision=info.sha,
        filename=filename,size=entry.size,sha256=checksum.hexdigest()),indent=2)+'\n',encoding='utf-8')
    if args.backend == 'pet_dino':
        language_revision = HfApi(token=False).model_info('google-bert/bert-base-uncased').sha
        snapshot_download('google-bert/bert-base-uncased',revision=language_revision,token=False,local_dir=target / 'bert-base-uncased',
                          allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                          'special_tokens_map.json', 'vocab.txt', 'model.safetensors'])
    else:
        # The official repository already ships the exact XLM-R tokenizer/config;
        # its trained text encoder weights are part of the detector checkpoint.
        language = 'xlm-roberta-large' if args.variant == 'large' else 'xlm-roberta-base'
        source = ROOT / 'third_party/wedetect' / language
        output = target / language
        output.mkdir(exist_ok=True)
        for name in ('config.json', 'tokenizer.json', 'sentencepiece.bpe.model'):
            shutil.copyfile(source / name, output / name)
        print(output)


if __name__ == '__main__':
    main()
