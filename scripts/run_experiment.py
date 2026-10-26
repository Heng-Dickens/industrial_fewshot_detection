"""Run a config-defined experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrial_fewshot.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
