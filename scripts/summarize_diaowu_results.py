"""Compatibility entry point for the unified per-dataset experiment table."""
import argparse
from pathlib import Path

try:
    from .summarize_results import refresh
except ImportError:
    from summarize_results import refresh

ROOT = Path(__file__).resolve().parents[1]


def main(root):
    root = Path(root).resolve()
    return refresh(root.parent.parent, [root.name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/runs/diaowu_val300')
    main(parser.parse_args().root)
