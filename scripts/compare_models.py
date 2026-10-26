"""Compare two or more saved detector runs on identical image bytes and GT."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrial_fewshot.evaluation.compare import main
if __name__ == "__main__":
    raise SystemExit(main())
