"""Restore only the original Qwen provider API-key field after explicit approval.

No requests, provider switching, model edits, Codex login edits or token copying.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

PROVIDER = '千问云'
UPSTREAM = 'https://maas.qianwenaiapi.com/compatible-mode/v1'


def restore_key(database: Path, backup: Path):
    if database.resolve() == backup.resolve():
        raise ValueError('目标数据库和只读备份不能相同')
    with closing(sqlite3.connect(backup.resolve().as_uri() + '?mode=ro', uri=True)) as source:
        rows = source.execute('SELECT id,settings_config FROM providers WHERE app_type=? AND name=?',
                              ('codex', PROVIDER)).fetchall()
    if len(rows) != 1:
        raise ValueError('备份中必须有且只有一个原千问云供应商')
    provider_id, encoded = rows[0]
    saved = json.loads(encoded)
    if 'base_url = "' + UPSTREAM + '"' not in saved.get('config', ''):
        raise ValueError('备份供应商上游与原实验不一致')
    key = saved.get('auth', {}).get('OPENAI_API_KEY')
    if not isinstance(key, str) or not key.strip() or key.strip() == 'EMPTY':
        raise ValueError('备份中缺少有效的原供应商 API Key')
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=rw', uri=True, timeout=10)) as target:
        with target:
            target.execute('BEGIN IMMEDIATE')
            row = target.execute('SELECT name,settings_config FROM providers WHERE app_type=? AND id=?',
                                 ('codex', provider_id)).fetchone()
            if row is None or row[0] != PROVIDER:
                raise ValueError('目标供应商身份与备份不一致')
            current = json.loads(row[1])
            if 'base_url = "' + UPSTREAM + '"' not in current.get('config', ''):
                raise ValueError('目标供应商地址已变化，未恢复密钥')
            auth = current.get('auth')
            if not isinstance(auth, dict):
                raise ValueError('目标供应商 auth 结构不符合预期')
            if auth.get('OPENAI_API_KEY'):
                return False
            auth['OPENAI_API_KEY'] = key
            target.execute('UPDATE providers SET settings_config=? WHERE app_type=? AND id=?',
                           (json.dumps(current, ensure_ascii=False), 'codex', provider_id))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--confirm-restore-key', action='store_true', required=True)
    args = parser.parse_args()
    changed = restore_key(args.database, args.backup)
    print(json.dumps({'provider': PROVIDER, 'changed': changed,
                      'changed_fields': ['auth.OPENAI_API_KEY'] if changed else [],
                      'existing_key_preserved': not changed}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
