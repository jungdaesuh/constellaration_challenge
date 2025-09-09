import sys
from pathlib import Path

# Ensure src directory is on sys.path for imports without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
