"""Make the helpers next to the docs tests importable as top-level modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
