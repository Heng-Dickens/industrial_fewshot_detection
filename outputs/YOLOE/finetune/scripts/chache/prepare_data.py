"""Task-specific entry point; shared implementation receives this task's config."""
from pathlib import Path
import sys

COMMON = Path(__file__).resolve().parents[1] / 'common'
sys.path.insert(0, str(COMMON))
from prepare_data import main as run_shared


if __name__ == '__main__':
    run_shared(default_config=Path(__file__).with_name('config.json'))
