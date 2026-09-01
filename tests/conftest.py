import sys
from pathlib import Path

# Let `pytest` work from a clean checkout without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
