"""v00: veri bozulmadan saglikli referans surumu hazirlar."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from .dosyalar import train_labels_sha256


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

