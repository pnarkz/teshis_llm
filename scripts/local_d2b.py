"""Run D2b missing-label experiment on the local GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Windows'ta symlink yetkisi gerektirmemesi icin yerel kosuda kopyala.
    shutil.copy2(source, target)


def build_dataset(source: Path, output: Path, drop_ratio: float, seed: int) -> Path:
    rng = random.Random(seed)
    label_dir = source / "labels/train"
    image_dir = source / "images/train"
    output.mkdir(parents=True, exist_ok=True)
    removed_by_class = {name: 0 for name in NAMES.values()}
    kept_rows = removed_rows = image_count = 0
    source_hash = hashlib.sha256()

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_image(image_dir, label_path.stem)
        if image_path is None:
            continue
        link_or_copy(image_path, output / "images/train" / image_path.name)
        image_count += 1
        source_hash.update(label_path.read_bytes())
        kept = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            fields = raw.split()
            if not raw.strip() or len(fields) != 5:
                kept.append(raw)
                continue
            class_id = int(fields[0])
            if rng.random() < drop_ratio:
                removed_rows += 1
                name = NAMES.get(class_id, str(class_id))
                removed_by_class[name] = removed_by_class.get(name, 0) + 1
            else:
                kept.append(raw)
                kept_rows += 1
        target = output / "labels/train" / label_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        f"val: {(source / 'images/val').resolve().as_posix()}\n"
        f"test: {(source / 'images/test').resolve().as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "dataset_manifest_v1",
        "scenario": "D2b",
        "version": "v03_d2b_eksik_etiket",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source.resolve()),
        "source_dataset_unchanged": True,
        "seed": seed,
        "parameters": {"box_drop_ratio": drop_ratio},
        "counts": {"train_images": image_count, "kept_bbox_rows": kept_rows, "removed_bbox_rows": removed_rows, "removed_by_class": removed_by_class},
        "source_train_labels_sha256": source_hash.hexdigest(),
        "val_test_modified": False,
        "files": {"data_yaml": str(data_yaml.resolve()), "output_dataset": str(output.resolve())},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GPU D2b training")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--model", default="main_model.pt", type=Path)
    parser.add_argument("--output-dataset", default="veri_surumleri/v03_d2b_eksik_etiket", type=Path)
    parser.add_argument("--output-root", default="experiments", type=Path)
    parser.add_argument("--run-name", default="run_D2b_42_local")
    parser.add_argument("--drop-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadi; bu kosu PC GPU ile yapilmalidir.")
    data_yaml = build_dataset(args.dataset.resolve(), args.output_dataset.resolve(), args.drop_ratio, args.seed)
    model = YOLO(str(args.model.resolve()))
    run_name = args.run_name
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
        device=0, workers=0, seed=args.seed, deterministic=True, cos_lr=True,
        patience=10, project=str(args.output_root.resolve()), name=run_name,
        exist_ok=True, plots=False, val=True, lr0=0.001, lrf=0.01,
        warmup_epochs=3, hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, degrees=5.0,
        translate=0.08, scale=0.3, shear=0.0, perspective=0.0, flipud=0.0,
        fliplr=0.5, mosaic=0.5, close_mosaic=8, mixup=0.0, copy_paste=0.0,
    )
    print(f"best_model={args.output_root.resolve() / run_name / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
