import sys
from pathlib import Path

# Bare `pytest` works in a fresh clone: the library source goes on the path
# here, before anything imports it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
