"""teshis paketinin pip ile kurulu olmasa bile testlerden import edilebilmesini saglar."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
