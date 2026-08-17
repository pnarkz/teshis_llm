"""Experiment metadata writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_manifest(output_dir: Path, values: dict[str, Any]) -> Path:
    """Write immutable run metadata next to Ultralytics outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "experiment_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
