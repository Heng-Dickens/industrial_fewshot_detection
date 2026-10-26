"""Check local Markdown link targets in README and docs, without network access."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def broken_links(root: Path) -> list[str]:
    errors = []
    for file in [root / 'README.md', root / 'configs/README.md', *sorted((root / 'docs').rglob('*.md'))]:
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


def main():
    errors = broken_links(Path(__file__).resolve().parents[1])
    print('\n'.join(errors) if errors else 'All local documentation links resolve.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
