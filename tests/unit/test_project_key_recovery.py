import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts import restore_qwen_project_key as recovery


class ProjectKeyRecoveryTest(unittest.TestCase):
    def test_only_api_key_is_copied_without_changing_source_or_printing_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup = root / 'fake-backup.db'
            with contextlib.closing(sqlite3.connect(backup)) as db:
                db.execute('CREATE TABLE providers (app_type TEXT, name TEXT, settings_config TEXT)')
                db.execute('INSERT INTO providers VALUES (?,?,?)', ('codex', '千问云', json.dumps({
                    'config': f'base_url = "{recovery.UPSTREAM}"',
                    'auth': {'OPENAI_API_KEY': 'FAKE-PROJECT-KEY', 'tokens': {'access_token': 'FAKE-CODEX-TOKEN'}}})))
                db.commit()
            before = backup.read_bytes()
            stdout = io.StringIO()
            with patch.object(recovery, 'ROOT', root), patch('sys.argv', ['recover', '--backup', str(backup)]), contextlib.redirect_stdout(stdout):
                self.assertEqual(recovery.main(), 0)
            saved = json.loads((root / 'configs/common/qwen_mllm.local.json').read_text())
            self.assertEqual(saved, {'api_key': 'FAKE-PROJECT-KEY'})
            self.assertEqual(backup.read_bytes(), before)
            self.assertNotIn('FAKE-PROJECT-KEY', stdout.getvalue())
            self.assertNotIn('FAKE-CODEX-TOKEN', stdout.getvalue())

    def test_existing_project_key_is_preserved_without_reading_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'configs/common/qwen_mllm.local.json'
            path.parent.mkdir(parents=True)
            path.write_text('{"api_key":"existing-private-key"}', encoding='utf-8')
            with patch.object(recovery, 'ROOT', root), patch('sys.argv', ['recover', '--backup', str(root / 'nonexistent.db')]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    recovery.main()
            self.assertEqual(json.loads(path.read_text())['api_key'], 'existing-private-key')
