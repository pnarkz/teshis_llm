"""Kaggle runner for D2a localization-noise experiment.

The input dataset is read-only. Train images are linked into /kaggle/working;
only copied train labels are modified. Validation and test labels remain in
the original input dataset, and test is never evaluated here.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
NOISE_RATIO = 0.15
SEED = 42
IMGSZ = 768
EPOCHS = 30
BATCH = 12


def find_input_dataset() -> Path:
    root = Path("/kaggle/input")
    candidates = []
    for image_dir in root.rglob("images/train"):
        dataset = image_dir.parent.parent
        if (dataset / "labels/train").is_dir():
            candidates.append(dataset)
    if not candidates:
        raise FileNotFoundError("/kaggle/input altinda images/train ve labels/train bulunamadi")
    return sorted(candidates, key=str)[0]


def find_input_model() -> Path:
    root = Path("/kaggle/input")
    preferred = list(root.rglob("main_model.pt"))
    if preferred:
        return sorted(preferred, key=str)[0]
    fallback = list(root.rglob("best.pt"))
    if fallback:
        return sorted(fallback, key=str)[0]
    raise FileNotFoundError("/kaggle/input altinda main_model.pt bulunamadi")


def find_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        path = image_dir / f"{stem}{extension}"
        if path.is_file():
            return path
    return None


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)


def noisy_label(raw: str, rng: random.Random) -> tuple[str, float, float] | None:
    fields = raw.split()
    if len(fields) != 5:
        return None
    try:
        class_id = int(fields[0])
        cx, cy, width, height = map(float, fields[1:])
    except ValueError:
        return None
    dx = rng.uniform(-NOISE_RATIO * width, NOISE_RATIO * width)
    dy = rng.uniform(-NOISE_RATIO * height, NOISE_RATIO * height)
    cx = min(max(cx + dx, width / 2), 1.0 - width / 2)
    cy = min(max(cy + dy, height / 2), 1.0 - height / 2)
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}", abs(dx), abs(dy)


def build_d2a(source: Path, output: Path) -> dict:
    train_image_dir = source / "images/train"
    train_label_dir = source / "labels/train"
    out_train_images = output / "images/train"
    out_train_labels = output / "labels/train"
    labels = sorted(train_label_dir.glob("*.txt"))
    if not labels:
        raise FileNotFoundError(f"Train etiketi bulunamadi: {train_label_dir}")

    rng = random.Random(SEED)
    changed_rows = 0
    invalid_rows = 0
    total_dx = 0.0
    total_dy = 0.0
    train_images = 0
    source_hash = hashlib.sha256()

    for label_path in labels:
        image_path = find_image(train_image_dir, label_path.stem)
        if image_path is None:
            continue
        link_or_copy(image_path, out_train_images / image_path.name)
        train_images += 1
        raw_lines = label_path.read_text(encoding="utf-8").splitlines()
        output_lines = []
        for raw in raw_lines:
            result = noisy_label(raw, rng)
            if result is None:
                if raw.strip():
                    invalid_rows += 1
                output_lines.append(raw)
                continue
            line, dx, dy = result
            output_lines.append(line)
            changed_rows += 1
            total_dx += dx
            total_dy += dy
        content = "\n".join(output_lines) + ("\n" if output_lines else "")
        (out_train_labels / label_path.name).parent.mkdir(parents=True, exist_ok=True)
        (out_train_labels / label_path.name).write_text(content, encoding="utf-8")
        source_hash.update(label_path.read_bytes())

    # Keep validation and test in the immutable input dataset.
    for split in ("val", "test"):
        (output / "images" / split).parent.mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).parent.mkdir(parents=True, exist_ok=True)

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
        "scenario": "D2a",
        "version": "v02_d2a_lokalizasyon_gurultusu",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_dataset_unchanged": True,
        "seed": SEED,
        "parameters": {"center_shift_ratio": NOISE_RATIO},
        "counts": {
            "train_images": train_images,
            "changed_bbox_rows": changed_rows,
            "invalid_rows_preserved": invalid_rows,
            "mean_abs_dx": total_dx / changed_rows if changed_rows else 0.0,
            "mean_abs_dy": total_dy / changed_rows if changed_rows else 0.0,
        },
        "source_train_labels_sha256": source_hash.hexdigest(),
        "val_test_modified": False,
        "files": {"data_yaml": str(data_yaml), "output_dataset": str(output)},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    from ultralytics import YOLO
    import torch

    source = find_input_dataset()
    model_path = find_input_model()
    work = Path("/kaggle/working")
    dataset = work / "v02_d2a_lokalizasyon_gurultusu"
    output = work / "experiments"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = build_d2a(source, dataset)
    print(json.dumps(manifest, indent=2))
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.train(
        data=str(dataset / "data.yaml"),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        device=device,
        workers=2,
        seed=SEED,
        deterministic=True,
        cos_lr=True,
        patience=10,
        project=str(output),
        name="run_D2a_42",
        exist_ok=True,
        plots=True,
        val=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=5.0,
        translate=0.08,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=8,
        mixup=0.0,
        copy_paste=0.0,
    )
    print(f"best_model={output / 'run_D2a_42/weights/best.pt'}")


if __name__ == "__main__":
    main()
