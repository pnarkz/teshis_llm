"""Deterministic dataset perturbations used by diagnostic scenarios."""

from __future__ import annotations

import random
from pathlib import Path


def label_class_ids(label_path: Path) -> set[int]:
    """Return class ids found in one YOLO label file."""
    ids: set[int] = set()
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        try:
            ids.add(int(fields[0]))
        except ValueError:
            continue
    return ids


def d1_remove_class_frames(
    label_paths: list[Path], class_id: int, remove_ratio: float, seed: int
) -> tuple[list[Path], dict[str, int]]:
    """Remove a seeded fraction of frames containing a target class."""
    if not 0 <= remove_ratio <= 1:
        raise ValueError("remove_ratio 0 ile 1 arasinda olmalidir")
    target = [p for p in label_paths if class_id in label_class_ids(p)]
    keep_target = round(len(target) * (1 - remove_ratio))
    selected = set(random.Random(seed).sample(target, keep_target))
    kept = [p for p in label_paths if class_id not in label_class_ids(p) or p in selected]
    return kept, {
        "train_frames_before": len(label_paths),
        "target_frames_before": len(target),
        "target_frames_kept": len(selected),
        "target_frames_removed": len(target) - len(selected),
        "train_frames_after": len(kept),
    }

def main() -> None:
    raise NotImplementedError("Bozulma fonksiyonlari senaryo YAML'larindan uretilir")
