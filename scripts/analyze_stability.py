"""Analyze variation across independent support seeds."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrial_fewshot.evaluation.fewshot_stability import main
if __name__ == "__main__":
    raise SystemExit(main())
