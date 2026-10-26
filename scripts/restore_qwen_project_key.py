"""Explicitly recover only the original project API key after user approval.

Reads a named CC Switch backup without changing it, writes an ignored project
credential file, and never prints credentials or sends a network request.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'https://maas.qianwenaiapi.com/compatible-mode/v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / 'configs/common'
    destination = directory / 'qwen_mllm.local.json'
    if (not directory.resolve().is_relative_to(ROOT.resolve())
            or not destination.resolve().is_relative_to(directory.resolve())):
        parser.error('Credential destination must stay inside project configs/common')
    if destination.exists():
        parser.error('Project credential file already exists; preserving it')
    with closing(sqlite3.connect(args.backup.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT settings_config FROM providers WHERE app_type=? AND name=?',
                          ('codex', '千问云')).fetchall()
    if len(rows) != 1:
        parser.error('Backup must contain exactly one matching original provider')
    settings = json.loads(rows[0][0])
    if 'base_url = "' + UPSTREAM + '"' not in settings.get('config', ''):
        parser.error('Backup provider does not match the original experiment upstream')
    key = settings.get('auth', {}).get('OPENAI_API_KEY')
    if not isinstance(key, str) or not key.strip() or key.strip() == 'EMPTY':
        parser.error('Backup does not contain an original provider API key')
    directory.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump({'api_key': key.strip()}, handle)
        handle.write('\n')
    print(f'Project-only API key saved to {destination}; shared settings unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
