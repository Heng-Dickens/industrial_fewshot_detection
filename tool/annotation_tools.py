"""Compatibility shim; no implementation is duplicated here."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from importlib import import_module
_module = import_module('industrial_fewshot.data.exports')
if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
