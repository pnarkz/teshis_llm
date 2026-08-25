"""Build and train the D3 UAP/UAI class-confusion experiment locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CLASS_NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def build_dataset(source: Path, output: Path, swap_ratio: float, seed: int) -> Path:
    if not 0 <= swap_ratio <= 1:
        raise ValueError("swap_ratio 0 ile 1 arasinda olmalidir")
    rng = random.Random(seed)
    source_labels = sorted((source / "labels/train").glob("*.txt"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "images/train").mkdir(parents=True, exist_ok=True)
    (output / "labels/train").mkdir(parents=True, exist_ok=True)
    rows = []
    source_hash = hashlib.sha256()
    image_count = missing_images = 0

    for label_path in source_labels:
        image_path = find_image(source / "images/train", label_path.stem)
        if image_path is None:
            missing_images += 1
            continue
        shutil.copy2(image_path, output / "images/train" / image_path.name)
        image_count += 1
        source_hash.update(label_path.read_bytes())
        for line_index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = raw.split()
            if len(fields) == 5 and fields[0] in {"2", "3"}:
                rows.append((label_path.name, line_index, raw))

    swap_count = round(len(rows) * swap_ratio)
    selected = set(rng.sample(range(len(rows)), swap_count))
    row_ids = {key: index for index, key in enumerate(rows)}
    changed_rows = 0
    changed_by_class = {"UAP_to_UAI": 0, "UAI_to_UAP": 0}

    for label_path in source_labels:
        image_path = find_image(source / "images/train", label_path.stem)
        if image_path is None:
            continue
        output_label = output / "labels/train" / label_path.name
        output_label.parent.mkdir(parents=True, exist_ok=True)
        changed = []
        for line_index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = raw.split()
            if len(fields) == 5 and fields[0] in {"2", "3"}:
                row_index = row_ids[(label_path.name, line_index, raw)]
                if row_index in selected:
                    old_id = fields[0]
                    fields[0] = "3" if old_id == "2" else "2"
                    changed_rows += 1
                    changed_by_class["UAP_to_UAI" if old_id == "2" else "UAI_to_UAP"] += 1
                    raw = " ".join(fields)
            changed.append(raw)
        output_label.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        f"val: {(Path('val_diagnostic') / 'images').resolve().as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "dataset_manifest_v1",
        "scenario": "D3",
        "version": "v04_d3_uap_uai_sinif_karisikligi",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source.resolve()),
        "source_dataset_unchanged": True,
        "seed": seed,
        "parameters": {"swap_ratio": swap_ratio, "class_ids": [2, 3]},
        "counts": {
            "train_images": image_count,
            "missing_images": missing_images,
            "uap_uai_rows_before": len(rows),
            "changed_rows": changed_rows,
            "changed_by_class": changed_by_class,
        },
        "val_diagnostic_modified": False,
        "source_train_labels_sha256": source_hash.hexdigest(),
        "files": {"data_yaml": str(data_yaml.resolve()), "output_dataset": str(output.resolve())},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GPU D3 training")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--model", default="main_model.pt", type=Path)
    parser.add_argument("--output-dataset", default="veri_surumleri/v04_d3_uap_uai_sinif_karisikligi", type=Path)
    parser.add_argument("--output-root", default="experiments", type=Path)
    parser.add_argument("--swap-ratio", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadi; D3 yerel GPU ile yapilmalidir.")
    data_yaml = build_dataset(args.dataset.resolve(), args.output_dataset.resolve(), args.swap_ratio, args.seed)
    model = YOLO(str(args.model.resolve()))
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
        device=0, workers=0, seed=args.seed, deterministic=True, cos_lr=True,
        patience=10, project=str(args.output_root.resolve()), name="run_D3_42_local",
        exist_ok=True, plots=False, val=True, lr0=0.001, lrf=0.01,
        warmup_epochs=3, hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, degrees=5.0,
        translate=0.08, scale=0.3, shear=0.0, perspective=0.0, flipud=0.0,
        fliplr=0.5, mosaic=0.5, close_mosaic=8, mixup=0.0, copy_paste=0.0,
    )
    print(f"best_model={args.output_root.resolve() / 'run_D3_42_local' / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
