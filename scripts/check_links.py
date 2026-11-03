"""Check local documentation links and optional output-table structure, offline."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def broken_links(root: Path, *, outputs_only=False, include_history=False) -> list[str]:
    errors = []
    files = [] if outputs_only else [root/'README.md', root/'configs/README.md']
    for directory in (['outputs'] if outputs_only else ['docs', 'outputs']):
        files.extend(sorted((root/directory).rglob('*.md')))
    for file in files:
        if not include_history and any(part in {'history', 'runtime', 'cache', '__pycache__', 'layout_indexes', 'summary_snapshots'} for part in file.relative_to(root).parts):
            continue
        if not file.is_file():
            continue
        text = file.read_text(encoding='utf-8-sig')
        for match in re.finditer(r'\[[^\]\n]*\]\(([^)\n]+)\)', text):
            target = match.group(1).split(' "')[0].strip('<>')
            if not target or target.startswith('#') or urlsplit(target).scheme:
                continue
            if not (file.parent / unquote(target.split('#')[0])).exists():
                line = text.count('\n', 0, match.start()) + 1
                errors.append(f'{file.relative_to(root)}:{line}: {target}')
    return errors


def output_tables(root: Path):
    """Check all Markdown/CSV tables, including the immutable historical copies."""
    counts = Counter()
    errors = []
    separator = re.compile(r'\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*')
    for file in sorted((root/'outputs').rglob('*.md')):
        counts['markdown_files'] += 1
        lines = file.read_text(encoding='utf-8-sig').splitlines()
        index = 0
        while index < len(lines)-1:
            if not lines[index].strip().startswith('|') or not separator.fullmatch(lines[index+1]):
                index += 1
                continue
            start = index
            rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                rows.append(lines[index].strip())
                index += 1
            counts['markdown_tables'] += 1
            width = len(re.split(r'(?<!\\)\|', rows[0].strip('|')))
            for offset, row in enumerate(rows):
                if len(re.split(r'(?<!\\)\|', row.strip('|'))) != width:
                    errors.append(f'{file.relative_to(root)}:{start+offset+1}: inconsistent table width')
            if len(rows[2:]) != len(set(rows[2:])):
                errors.append(f'{file.relative_to(root)}:{start+1}: duplicate table rows')
    for file in sorted((root/'outputs').rglob('*.csv')):
        counts['csv_files'] += 1
        with file.open(encoding='utf-8-sig', newline='') as stream:
            rows = [r for r in csv.reader(stream) if any(r)]
        if not rows:
            continue
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            errors.append(f'{file.relative_to(root)}: inconsistent CSV width')
        if len(rows[1:]) != len(set(map(tuple, rows[1:]))):
            errors.append(f'{file.relative_to(root)}: duplicate CSV rows')
    return dict(counts), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-only', action='store_true')
    parser.add_argument('--tables', action='store_true', help='Also check every output Markdown/CSV table.')
    parser.add_argument('--include-history', action='store_true', help='Include original links in immutable historical snapshots.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = broken_links(root, outputs_only=args.outputs_only, include_history=args.include_history)
    if args.tables:
        counts, table_errors = output_tables(root)
        errors.extend(table_errors)
        print('Output table coverage:', counts)
    print('\n'.join(errors) if errors else 'All local documentation links resolve.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
