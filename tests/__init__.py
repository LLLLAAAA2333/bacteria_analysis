"""Make the two source trees available to the unified test suite."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "poster" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
