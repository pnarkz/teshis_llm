"""Kaggle runner for D2b missing-label experiment."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

from kaggle_d2a import find_image, find_input_dataset, find_input_model, link_or_copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teshis.egitim.protokol import egitim_kwargs  # noqa: E402


SEED = 42
DROP_RATIO = 0.25
IMGSZ = 768
EPOCHS = 30
BATCH = 12
NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}


def build_d2b(source: Path, output: Path) -> dict:
    train_image_dir = source / "images/train"
    train_label_dir = source / "labels/train"
    out_image_dir = output / "images/train"
    out_label_dir = output / "labels/train"
    rng = random.Random(SEED)
    source_hash = hashlib.sha256()
    removed_by_class = {name: 0 for name in NAMES.values()}
    kept_rows = 0
    removed_rows = 0
    train_images = 0

    for label_path in sorted(train_label_dir.glob("*.txt")):
        image_path = find_image(train_image_dir, label_path.stem)
        if image_path is None:
            continue
        link_or_copy(image_path, out_image_dir / image_path.name)
        train_images += 1
        source_hash.update(label_path.read_bytes())
        kept = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            fields = raw.split()
            if not raw.strip() or len(fields) != 5:
                kept.append(raw)
                continue
            class_id = int(fields[0])
            if rng.random() < DROP_RATIO:
                removed_rows += 1
                removed_by_class[NAMES.get(class_id, str(class_id))] = removed_by_class.get(NAMES.get(class_id, str(class_id)), 0) + 1
            else:
                kept.append(raw)
                kept_rows += 1
        target = out_label_dir / label_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        f"val: {(source / 'images/val').as_posix()}\n"
        f"test: {(source / 'images/test').as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "dataset_manifest_v1",
        "scenario": "D2b",
        "version": "v03_d2b_eksik_etiket",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_dataset_unchanged": True,
        "seed": SEED,
        "parameters": {"box_drop_ratio": DROP_RATIO},
        "counts": {
            "train_images": train_images,
            "kept_bbox_rows": kept_rows,
            "removed_bbox_rows": removed_rows,
            "removed_by_class": removed_by_class,
        },
        "source_train_labels_sha256": source_hash.hexdigest(),
        "val_test_modified": False,
        "files": {"data_yaml": str(data_yaml), "output_dataset": str(output)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    source = find_input_dataset()
    model_path = find_input_model()
    output_root = Path("/kaggle/working")
    dataset = output_root / "v03_d2b_eksik_etiket"
    experiment_root = output_root / "experiments"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = build_d2b(source, dataset)
    print(json.dumps(manifest, indent=2))
    model = YOLO(str(model_path))
    model.train(
        data=str(dataset / "data.yaml"),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        seed=SEED,
        project=str(experiment_root),
        name="run_D2b_42",
        exist_ok=True,
        plots=True,
        val=True,
        **egitim_kwargs(),
    )
    print(f"best_model={experiment_root / 'run_D2b_42/weights/best.pt'}")


if __name__ == "__main__":
    main()
