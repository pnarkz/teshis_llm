"""Create reproducible, manifest-only dataset versions for scenarios."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .bozulmalar import d1_remove_class_frames


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 veri surumu uret")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", default="senaryolar/veri/d1_sinif_yetersizligi.yaml", type=Path)
    parser.add_argument("--output-root", default="veri_surumleri", type=Path)
    args = parser.parse_args()
    manifest = build_d1(args.dataset.resolve(), args.output_root.resolve(), args.config.resolve())
    print(f"D1 manifest: {manifest}")


if __name__ == "__main__":
    main()
