"""Evaluate a completed Diaowu training run on the frozen 300-image test set."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from evaluate import main

if __name__ == '__main__':
    main(default_config=Path(__file__).with_name('config.json'))
