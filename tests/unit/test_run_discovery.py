from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from industrial_fewshot.run_discovery import saved_runs


class RunDiscoveryTest(unittest.TestCase):
    def test_unreadable_history_does_not_hide_current_dataset_prompt_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / 'dataset/prompt_v2/model_5shot/reference_group'
            leaf.mkdir(parents=True)
            metadata = leaf / 'run.json'
            metadata.write_text('{}', encoding='utf-8')
            inaccessible = root / 'unavailable_history'
            inaccessible.mkdir()
            cache = root / '.inference_cache'
            cache.mkdir()
            (cache / 'run.json').write_text('{}', encoding='utf-8')
            original = Path.iterdir
            def iterdir(path):
                if path == inaccessible:
                    raise PermissionError('historical directory unavailable')
                return original(path)
            with patch.object(Path, 'iterdir', iterdir):
                self.assertEqual(list(saved_runs(root)), [metadata])
