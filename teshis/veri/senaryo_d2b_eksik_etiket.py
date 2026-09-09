"""D2b: egitim etiketlerinden seed ile belirlenen kutulari siler.

build_dataset yerel, build_d2b tarihsel Kaggle manifest bicimini korur.
Bu modul egitim baslatmaz; komut girisleri scripts/ altindadir.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .senaryo_d2a_lokalizasyon_gurultusu import link_or_copy as link_or_copy_kaggle


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


# Kaggle girisinin tarihsel manifest ve baglanti davranisi.

SEED = 42
DROP_RATIO = 0.25


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
        link_or_copy_kaggle(image_path, out_image_dir / image_path.name)
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

