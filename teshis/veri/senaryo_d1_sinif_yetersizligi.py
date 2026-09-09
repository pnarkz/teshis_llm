"""D1: hedef sinifi iceren egitim karelerinin bir bolumunu cikarir."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import random
import yaml
from .dosyalar import find_image


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


def build_d1(dataset_root: Path, output_root: Path, config_path: Path) -> Path:
    """Build D1 without copying source images or changing source labels."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["parametreler"]
    class_id = int(params["sinif_id"])
    remove_ratio = float(params["kare_cikarma_orani"])
    seed = int(config.get("seed", 42))

    train_labels_dir = dataset_root / "labels" / "train"
    train_images_dir = dataset_root / "images" / "train"
    labels = sorted(train_labels_dir.glob("*.txt"))
    if not labels:
        raise FileNotFoundError(f"Train label bulunamadi: {train_labels_dir}")

    labels_with_images = [p for p in labels if find_image(train_images_dir, p.stem)]
    kept_labels, counts = d1_remove_class_frames(labels_with_images, class_id, remove_ratio, seed)
    missing_images = len(labels) - len(labels_with_images)

    version_root = output_root / "v01_d1_sinif_yetersizligi"
    version_root.mkdir(parents=True, exist_ok=True)
    train_list = version_root / "train_images.txt"
    train_list.write_text(
        "\n".join(str(find_image(train_images_dir, p.stem).resolve()) for p in kept_labels) + "\n",
        encoding="utf-8",
    )

    val_images = (dataset_root / "images" / "val").resolve()
    test_images = (dataset_root / "images" / "test").resolve()
    data_yaml = version_root / "data.yaml"
    data_yaml.write_text(
        "# D1: only the train image list is reduced; val/test remain operational.\n"
        f"path: {dataset_root.resolve().as_posix()}\n"
        f"train: {train_list.resolve().as_posix()}\n"
        f"val: {val_images.as_posix()}\n"
        f"test: {test_images.as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )

    manifest = {
        "format": "dataset_manifest_v1",
        "version": "v01_d1_sinif_yetersizligi",
        "scenario": "D1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_root.resolve()),
        "source_dataset_unchanged": True,
        "copy_mode": "manifest_only",
        "seed": seed,
        "target_class": {"id": class_id, "name": "insan" if class_id == 1 else str(class_id)},
        "parameters": {"frame_removal_ratio": remove_ratio},
        "counts": {**counts, "labels_without_images": missing_images},
        "files": {
            "data_yaml": str(data_yaml.resolve()),
            "train_images_list": str(train_list.resolve()),
            "val_source": str(val_images),
            "test_source": str(test_images),
        },
    }
    manifest_path = version_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path

