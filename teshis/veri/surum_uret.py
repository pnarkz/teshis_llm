"""Create reproducible, manifest-only dataset versions for scenarios."""

from __future__ import annotations

import argparse
import hashlib
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


def train_labels_sha256(dataset_root: Path) -> tuple[str, int, int]:
    """Train etiketlerinin birlesik sha256'sini, dosya ve bbox sayisini dondurur.

    Diger senaryo ureticileri (local_d2b, local_d3, kaggle_*) ayni alani
    manifest'e yaziyor. v00 icin de kaydedilince, tum bozuk surumlerin ayni
    kaynaktan turedigi hash karsilastirmasiyla ispatlanabilir hale gelir.
    """
    digest = hashlib.sha256()
    dosya = bbox = 0
    for label_path in sorted((dataset_root / "labels" / "train").iterdir()):
        if label_path.suffix != ".txt":
            continue
        veri = label_path.read_bytes()
        digest.update(veri)
        dosya += 1
        bbox += sum(1 for satir in veri.decode("utf-8").splitlines() if len(satir.split()) >= 5)
    return digest.hexdigest(), dosya, bbox


def build_v00(dataset_root: Path, output_root: Path) -> Path:
    """Saglikli referans surumu: kaynak dataset hic degistirilmez.

    katalog.yaml ve tum senaryo configleri `kaynak_surum: v00_saglikli` diyor.
    Bu surum, bozulmus surumlerle AYNI protokolde egitilecek temiz referansi
    tanimlar. Onceden boyle bir surum yoktu ve karsilastirmalar hic fine-tune
    edilmemis main_model.pt'ye karsi yapiliyordu; bu, "bozulma etkisi" ile
    "fine-tune etkisi"ni birbirine karistiriyordu.
    """
    for alt in ("images/train", "labels/train", "images/val", "images/test"):
        if not (dataset_root / alt).is_dir():
            raise FileNotFoundError(f"Kaynak dataset eksik: {dataset_root / alt}")

    version_root = output_root / "v00_saglikli"
    version_root.mkdir(parents=True, exist_ok=True)
    hash_degeri, dosya, bbox = train_labels_sha256(dataset_root)

    data_yaml = version_root / "data.yaml"
    data_yaml.write_text(
        "# v00: saglikli referans. Kaynak dataset hicbir sekilde degistirilmez;\n"
        "# bu surum yalnizca ortak protokolle egitilecek temiz referansi tanimlar.\n"
        f"path: {dataset_root.resolve().as_posix()}\n"
        f"train: {(dataset_root / 'images/train').resolve().as_posix()}\n"
        f"val: {(dataset_root / 'images/val').resolve().as_posix()}\n"
        f"test: {(dataset_root / 'images/test').resolve().as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "dataset_manifest_v1",
        "version": "v00_saglikli",
        "scenario": "v00",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_root.resolve()),
        "source_dataset_unchanged": True,
        "copy_mode": "manifest_only",
        "bozulma": None,
        "counts": {"train_label_files": dosya, "train_bbox": bbox},
        "source_train_labels_sha256": hash_degeri,
        "files": {"data_yaml": str(data_yaml.resolve())},
    }
    manifest_path = version_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


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
    parser = argparse.ArgumentParser(description="Veri surumu uret (v00 saglikli veya D1)")
    parser.add_argument("--surum", choices=("v00", "d1"), default="d1")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", default="senaryolar/veri/d1_sinif_yetersizligi.yaml", type=Path)
    parser.add_argument("--output-root", default="veri_surumleri", type=Path)
    args = parser.parse_args()
    if args.surum == "v00":
        manifest = build_v00(args.dataset.resolve(), args.output_root.resolve())
        print(f"v00 manifest: {manifest}")
        return
    manifest = build_d1(args.dataset.resolve(), args.output_root.resolve(), args.config.resolve())
    print(f"D1 manifest: {manifest}")


if __name__ == "__main__":
    main()
