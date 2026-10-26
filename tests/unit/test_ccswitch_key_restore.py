from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.restore_ccswitch_qwen_key import PROVIDER, UPSTREAM, restore_key


class CCSwitchKeyRestoreTest(unittest.TestCase):
    def create_db(self, path, key):
        settings = {'config': f'model = "gpt-6.1-sol"\nbase_url = "{UPSTREAM}"',
                    'auth': {'OPENAI_API_KEY': key, 'tokens': {'access_token': 'FAKE-LOGIN-TOKEN'},
                             'auth_mode': 'chatgpt'}, 'other': {'keep': True}}
        with closing(sqlite3.connect(path)) as db:
            db.execute('CREATE TABLE providers (id TEXT,app_type TEXT,name TEXT,settings_config TEXT,is_current INTEGER)')
            db.execute('INSERT INTO providers VALUES (?,?,?,?,?)',
                       ('test-provider', 'codex', PROVIDER, json.dumps(settings), 1))
            db.commit()
        return settings

    def read(self, path):
        with closing(sqlite3.connect(path)) as db:
            row = db.execute('SELECT settings_config,is_current FROM providers').fetchone()
        return json.loads(row[0]), row[1]

    def test_only_missing_key_is_restored_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'backup.db', root / 'current.db'
            self.create_db(source, 'FAKE-ORIGINAL-KEY')
            current = self.create_db(target, None)
            original_bytes = source.read_bytes()
            self.assertTrue(restore_key(target, source))
            expected = copy.deepcopy(current)
            expected['auth']['OPENAI_API_KEY'] = 'FAKE-ORIGINAL-KEY'
            self.assertEqual(self.read(target), (expected, 1))
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_existing_user_key_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'backup.db', root / 'current.db'
            self.create_db(source, 'FAKE-OLD-KEY')
            current = self.create_db(target, 'FAKE-CURRENT-KEY')
            self.assertFalse(restore_key(target, source))
            self.assertEqual(self.read(target), (current, 1))

    def test_changed_upstream_is_rejected_without_target_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'backup.db', root / 'current.db'
            self.create_db(source, 'FAKE-KEY')
            current = self.create_db(target, None)
            current['config'] = 'base_url = "https://different.example/v1"'
            with closing(sqlite3.connect(target)) as db:
                db.execute('UPDATE providers SET settings_config=?', (json.dumps(current),))
                db.commit()
            with self.assertRaisesRegex(ValueError, '地址已变化'):
                restore_key(target, source)
            self.assertEqual(self.read(target), (current, 1))
