"""Operasyonel val'den kaynak-tekil tanisal val uretir."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from teshis.config import aktif_ortam, yukle
from teshis.veri.istatistik import (
    GORUNTU_UZANTILARI,
    SINIFLAR,
    kaynak_adi,
    kaynak_govde,
)

SPLITLER = ("train", "val", "test")


def _dosyalar(root: Path, split: str) -> dict[str, Path]:
    return {
        dosya.stem: dosya
        for dosya in (root / "images" / split).iterdir()
        if dosya.suffix.lower() in GORUNTU_UZANTILARI
    }


def _sinif_say(etiket: Path) -> Counter[str]:
    sayac: Counter[str] = Counter()
    for satir in etiket.read_text(encoding="utf-8").splitlines():
        kolonlar = satir.split()
        if not kolonlar:
            continue
        try:
            sinif_id = int(float(kolonlar[0]))
        except ValueError:
            continue
        if sinif_id in SINIFLAR:
            sayac[SINIFLAR[sinif_id]] += 1
    return sayac


def _checksum(yol: Path) -> str:
    sha = hashlib.sha256()
    with yol.open("rb") as dosya:
        for parca in iter(lambda: dosya.read(1024 * 1024), b""):
            sha.update(parca)
    return sha.hexdigest()


def olustur(dataset_root: Path, output_root: Path, seed: int = 42) -> dict[str, Any]:
    """Tanisal val klasoru ve manifest'i uretir."""
    images = output_root / "images"
    labels = output_root / "labels"
    output_root.mkdir(parents=True, exist_ok=True)
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)

    split_files = {split: _dosyalar(dataset_root, split) for split in SPLITLER}
    val_files = split_files["val"]
    train_keys = {kaynak_govde(path.name) for path in split_files["train"].values()}
    test_keys = {kaynak_govde(path.name) for path in split_files["test"].values()}

    gruplar: dict[str, list[tuple[str, Path]]] = {}
    dislananlar: list[dict[str, str]] = []
    for stem, image_path in val_files.items():
        kaynak = kaynak_adi(image_path.name)
        kaynak_key = kaynak_govde(image_path.name)
        if kaynak == "sentetik":
            dislananlar.append({"stem": stem, "neden": "sentetik"})
            continue
        if kaynak_key in train_keys:
            dislananlar.append({"stem": stem, "neden": "train_kaynak_tekrari"})
            continue
        if kaynak_key in test_keys:
            dislananlar.append({"stem": stem, "neden": "test_kaynak_tekrari"})
            continue
        gruplar.setdefault(kaynak_key, []).append((stem, image_path))

    rng = random.Random(seed)
    secilenler: list[tuple[str, Path]] = []
    varyant_sayilari: Counter[str] = Counter()
    for kaynak_key, adaylar in sorted(gruplar.items()):
        rng.shuffle(adaylar)
        secilenler.append(adaylar[0])
        varyant_sayilari[str(len(adaylar))] += 1

    secilenler.sort(key=lambda item: item[0])
    sinif_bbox: Counter[str] = Counter()
    kaynak_bbox: dict[str, Counter[str]] = {}
    dosyalar: list[dict[str, Any]] = []
    for stem, image_path in secilenler:
        label_path = dataset_root / "labels" / "val" / f"{stem}.txt"
        if not label_path.exists():
            dislananlar.append({"stem": stem, "neden": "etiket_eksik"})
            continue
        target_image = images / image_path.name
        target_label = labels / label_path.name
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        kaynak = kaynak_adi(image_path.name)
        kutular = _sinif_say(label_path)
        sinif_bbox.update(kutular)
        kaynak_bbox.setdefault(kaynak, Counter()).update(kutular)
        dosyalar.append({
            "image": image_path.name,
            "label": label_path.name,
            "source_group": kaynak,
            "source_key": kaynak_govde(image_path.name),
            "image_sha256": _checksum(image_path),
        })

    manifest: dict[str, Any] = {
        "format": "val_diagnostic_manifest_v1",
        "amac": "kaynak_tekil_ve_split_ayrik_tanisal_val",
        "kaynak_dataset": str(dataset_root.resolve()),
        "cikti": str(output_root.resolve()),
        "seed": seed,
        "kaynak_split": "val",
        "sentetik_dahil": False,
        "train_test_kaynak_tekrari_dahil": False,
        "goruntu_sayisi": len(dosyalar),
        "bbox_sayisi": sum(sinif_bbox.values()),
        "sinif_bbox": dict(sinif_bbox),
        "kaynak_grubu": {
            kaynak: {"goruntu": sum(1 for dosya in dosyalar if dosya["source_group"] == kaynak),
                     "bbox": sum(sayilar.values())}
            for kaynak, sayilar in kaynak_bbox.items()
        },
        "roboflow_varyant_gruplari": dict(varyant_sayilari),
        "dislanan_sayi": len(dislananlar),
        "dislanan_ornekler": dislananlar[:100],
        "dosyalar": dosyalar,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    data_yaml = (
        f"path: {output_root.as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Tanisal val uret")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = yukle(args.config)
    ortam = aktif_ortam(config)
    seed = config.get("tohum", 42) if args.seed is None else args.seed
    output_root = Path(ortam["output_root"]) / "val_diagnostic"
    manifest = olustur(Path(ortam["dataset_root"]), output_root, seed)
    print(f"val_diagnostic goruntu: {manifest['goruntu_sayisi']}")
    print(f"val_diagnostic bbox: {manifest['bbox_sayisi']}")
    print(f"sinif bbox: {manifest['sinif_bbox']}")
    print(f"dislanan: {manifest['dislanan_sayi']}")
    print(f"manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
