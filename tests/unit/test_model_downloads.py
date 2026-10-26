import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import download_prompt_detector_weights as native
from scripts import download_groundingdino as grounding


class ModelDownloadTests(unittest.TestCase):
    def test_wedetect_large_language_and_existing_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'weights/wedetect'
            target.mkdir(parents=True)
            checkpoint=target/'wedetect_large.pth'
            checkpoint.write_bytes(b'official-test-checkpoint')
            language=root/'third_party/wedetect/xlm-roberta-large'
            language.mkdir(parents=True)
            for name in ('config.json','tokenizer.json','sentencepiece.bpe.model'):
                (language/name).write_bytes(b'official-test-language')
            entry=SimpleNamespace(rfilename='wedetect_large.pth',size=checkpoint.stat().st_size,
                                  lfs=SimpleNamespace(sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            hub=SimpleNamespace(HfApi=lambda **_:SimpleNamespace(model_info=lambda *a,**k:SimpleNamespace(sha='test-commit',siblings=[entry])),
                                hf_hub_download=lambda *a,**k:self.fail('Must preserve existing checkpoint'),
                                snapshot_download=lambda *a,**k:self.fail('Text weights are in the checkpoint'))
            with patch.dict(sys.modules,{'huggingface_hub':hub}),patch.object(native,'ROOT',root),patch.object(sys,'argv',['download','--backend','wedetect','--variant','large']):
                native.main()
                self.assertEqual((target/'xlm-roberta-large/tokenizer.json').read_bytes(),b'official-test-language')
                self.assertEqual(json.loads(checkpoint.with_suffix('.manifest.json').read_text())['revision'],'test-commit')
                checkpoint.write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError,'not overwritten'):
                    native.main()
                self.assertEqual(checkpoint.read_bytes(),b'corrupt')

    def test_grounding_base_revision_and_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def download(repo,**kwargs):
                self.assertEqual(repo,'IDEA-Research/grounding-dino-base')
                self.assertEqual(kwargs['revision'],'fixed-test-commit')
                destination=Path(kwargs['local_dir'])
                destination.mkdir(parents=True)
                (destination/'config.json').write_text('{}')
                (destination/'model.safetensors').write_bytes(b'fixture')
            hub=SimpleNamespace(HfApi=lambda **_:SimpleNamespace(model_info=lambda *a,**k:SimpleNamespace(sha='fixed-test-commit')),
                                snapshot_download=download)
            with patch.dict(sys.modules,{'huggingface_hub':hub}),patch.object(grounding,'ROOT',root):
                grounding.main(['--variant','base'])
                (root/'weights/grounding-dino-base/model.safetensors').unlink()
                hub.snapshot_download=lambda *a,**k:None
                with self.assertRaisesRegex(RuntimeError,'incomplete'):
                    grounding.main(['--variant','base'])


if __name__=='__main__':
    unittest.main()
